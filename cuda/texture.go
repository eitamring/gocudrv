package cuda

import (
	"context"
	"math"
	"reflect"
	"runtime"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/argpack"
)

// AddressMode selects how a texture fetch treats coordinates outside the array.
// The values map 1:1 to the driver's CUaddress_mode. Wrap and Mirror take
// effect only with normalized coordinates; without them the driver clamps.
type AddressMode uint32

const (
	AddressWrap   AddressMode = AddressMode(cudasys.AddressModeWrap)
	AddressClamp  AddressMode = AddressMode(cudasys.AddressModeClamp)
	AddressMirror AddressMode = AddressMode(cudasys.AddressModeMirror)
	AddressBorder AddressMode = AddressMode(cudasys.AddressModeBorder)
)

// FilterMode selects how a texture fetch samples between elements. The values
// map 1:1 to the driver's CUfilter_mode. Linear filtering requires a float
// element type.
type FilterMode uint32

const (
	FilterPoint  FilterMode = FilterMode(cudasys.FilterModePoint)
	FilterLinear FilterMode = FilterMode(cudasys.FilterModeLinear)
)

// TextureConfig is the sampling configuration for NewTexture. The zero value is
// point sampling with wrap addressing, which the driver treats as clamp unless
// NormalizedCoordinates is set.
type TextureConfig struct {
	AddressMode           AddressMode
	FilterMode            FilterMode
	NormalizedCoordinates bool
}

// Array2D is a CUDA array: a 2D device allocation in an opaque,
// texture-optimized layout with no device pointer, the memory a Texture samples
// from. Close it after any Texture over it and before the owning Context.
type Array2D[T Supported] struct {
	ctx    *Context
	handle cudasys.CUarray
	width  int
	height int
	opMu   sync.RWMutex
	closed bool
}

func arrayFormat[T Supported]() (format uint32, integer, ok bool) {
	switch reflect.TypeOf(*new(T)).Kind() {
	case reflect.Uint8:
		return cudasys.AdFormatUnsignedInt8, true, true
	case reflect.Uint16:
		return cudasys.AdFormatUnsignedInt16, true, true
	case reflect.Uint32:
		return cudasys.AdFormatUnsignedInt32, true, true
	case reflect.Int8:
		return cudasys.AdFormatSignedInt8, true, true
	case reflect.Int16:
		return cudasys.AdFormatSignedInt16, true, true
	case reflect.Int32:
		return cudasys.AdFormatSignedInt32, true, true
	case reflect.Float32:
		return cudasys.AdFormatFloat, false, true
	default:
		return 0, false, false
	}
}

// AllocArray2D creates a width-by-height CUDA array (in elements) with
// cuArrayCreate. 8-byte element types are rejected with ErrUnsupportedElement;
// a driver without the array symbols returns ErrSymbolUnavailable.
func AllocArray2D[T Supported](ctx *Context, width, height int) (*Array2D[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if width <= 0 || height <= 0 {
		return nil, ErrInvalidLength
	}
	if width > math.MaxInt/height {
		return nil, ErrInvalidLength
	}
	format, _, ok := arrayFormat[T]()
	if !ok {
		return nil, ErrUnsupportedElement
	}
	desc := cudasys.CUDA_ARRAY_DESCRIPTOR{
		Width:       uint64(width),
		Height:      uint64(height),
		Format:      format,
		NumChannels: 1,
	}

	var handle cudasys.CUarray
	err := ctx.do(context.Background(), func() error {
		h, e := cudaresult.ArrayCreate(ctx.driver, &desc)
		if e != nil {
			return e
		}
		handle = h
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Array2D[T]{ctx: ctx, handle: handle, width: width, height: height}, nil
}

// Width is the row length in elements. It is 0 for a nil receiver.
func (a *Array2D[T]) Width() int {
	if a == nil {
		return 0
	}
	return a.width
}

// Height is the number of rows. It is 0 for a nil receiver.
func (a *Array2D[T]) Height() int {
	if a == nil {
		return 0
	}
	return a.height
}

// Raw is the underlying CUarray handle for sibling libraries, a snapshot valid
// only while the array is open. It is 0 for a nil receiver.
func (a *Array2D[T]) Raw() cudasys.CUarray {
	if a == nil {
		return 0
	}
	return a.handle
}

func (a *Array2D[T]) hostDesc(hostToArray bool, host unsafe.Pointer) cudasys.Memcpy2D {
	widthBytes := uint64(a.width) * elemSize[T]()
	desc := cudasys.Memcpy2D{
		WidthInBytes: widthBytes,
		Height:       uint64(a.height),
	}
	if hostToArray {
		desc.SrcMemoryType, desc.SrcHost, desc.SrcPitch = cudasys.MemoryTypeHost, host, widthBytes
		desc.DstMemoryType, desc.DstArray = cudasys.MemoryTypeArray, uintptr(a.handle)
	} else {
		desc.SrcMemoryType, desc.SrcArray = cudasys.MemoryTypeArray, uintptr(a.handle)
		desc.DstMemoryType, desc.DstHost, desc.DstPitch = cudasys.MemoryTypeHost, host, widthBytes
	}
	return desc
}

// CopyFrom copies a packed host slice of Width*Height elements into the array
// with cuMemcpy2D. len(src) must equal Width*Height.
func (a *Array2D[T]) CopyFrom(ctx context.Context, src []T) error {
	if a == nil {
		return ErrNilArray
	}
	a.opMu.RLock()
	defer a.opMu.RUnlock()
	if a.closed {
		return ErrArrayClosed
	}
	if len(src) != a.width*a.height {
		return ErrLengthMismatch
	}
	desc := a.hostDesc(true, unsafe.Pointer(&src[0]))
	err := a.ctx.doWait(ctx, func() error {
		return cudaresult.Memcpy2D(a.ctx.driver, &desc)
	})
	runtime.KeepAlive(src)
	return err
}

// CopyTo copies the array into a packed host slice of Width*Height elements
// with cuMemcpy2D. len(dst) must equal Width*Height.
func (a *Array2D[T]) CopyTo(ctx context.Context, dst []T) error {
	if a == nil {
		return ErrNilArray
	}
	a.opMu.RLock()
	defer a.opMu.RUnlock()
	if a.closed {
		return ErrArrayClosed
	}
	if len(dst) != a.width*a.height {
		return ErrLengthMismatch
	}
	desc := a.hostDesc(false, unsafe.Pointer(&dst[0]))
	err := a.ctx.doWait(ctx, func() error {
		return cudaresult.Memcpy2D(a.ctx.driver, &desc)
	})
	runtime.KeepAlive(dst)
	return err
}

// Close destroys the array with cuArrayDestroy. Close every Texture created
// over it first. Idempotent; a failed destroy leaves the array open to retry.
func (a *Array2D[T]) Close() error {
	if a == nil {
		return ErrNilArray
	}
	a.opMu.Lock()
	defer a.opMu.Unlock()
	if a.closed {
		return nil
	}
	if err := a.ctx.doWait(context.Background(), func() error {
		return cudaresult.ArrayDestroy(a.ctx.driver, a.handle)
	}); err != nil {
		return err
	}
	a.closed = true
	return nil
}

// Texture is a sampling view over an Array2D, fetched through the texture
// cache with addressing, filtering, and optional coordinate normalization. Pass
// it to a kernel with ArgTexture; close it before the array it samples.
type Texture struct {
	ctx    *Context
	handle cudasys.CUtexObject
	opMu   sync.RWMutex
	closed bool
}

// NewTexture creates a texture object sampling arr with cfg; integer element
// types are read as integers. The array must stay open for the texture's
// lifetime. Returns ErrSymbolUnavailable on a driver without the symbols.
func NewTexture[T Supported](arr *Array2D[T], cfg TextureConfig) (*Texture, error) {
	if arr == nil {
		return nil, ErrNilArray
	}
	arr.opMu.RLock()
	defer arr.opMu.RUnlock()
	if arr.closed {
		return nil, ErrArrayClosed
	}
	_, integer, ok := arrayFormat[T]()
	if !ok {
		return nil, ErrUnsupportedElement
	}

	resDesc := cudasys.CUDA_RESOURCE_DESC{
		ResType: cudasys.ResourceTypeArray,
		Handle:  arr.handle,
	}
	texDesc := cudasys.CUDA_TEXTURE_DESC{
		AddressMode: [3]uint32{uint32(cfg.AddressMode), uint32(cfg.AddressMode), uint32(cfg.AddressMode)},
		FilterMode:  uint32(cfg.FilterMode),
	}
	if cfg.NormalizedCoordinates {
		texDesc.Flags |= cudasys.TexNormalizedCoordinate
	}
	if integer {
		texDesc.Flags |= cudasys.TexReadAsInteger
	}

	var handle cudasys.CUtexObject
	err := arr.ctx.do(context.Background(), func() error {
		h, e := cudaresult.TexObjectCreate(arr.ctx.driver, &resDesc, &texDesc)
		if e != nil {
			return e
		}
		handle = h
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Texture{ctx: arr.ctx, handle: handle}, nil
}

// Raw is the underlying CUtexObject handle (the value a kernel receives as a
// cudaTextureObject_t), a snapshot valid only while the texture is open. It is
// 0 for a nil receiver.
func (t *Texture) Raw() cudasys.CUtexObject {
	if t == nil {
		return 0
	}
	return t.handle
}

// Close destroys the texture object with cuTexObjectDestroy. Idempotent; a
// failed destroy leaves the texture open to retry.
func (t *Texture) Close() error {
	if t == nil {
		return ErrNilTexture
	}
	t.opMu.Lock()
	defer t.opMu.Unlock()
	if t.closed {
		return nil
	}
	if err := t.ctx.doWait(context.Background(), func() error {
		return cudaresult.TexObjectDestroy(t.ctx.driver, t.handle)
	}); err != nil {
		return err
	}
	t.closed = true
	return nil
}

type textureKernelArg struct {
	t *Texture
}

// ArgTexture passes a texture object to a kernel (a cudaTextureObject_t
// parameter). Like Arg it holds the texture's read lock across submission; the
// sampled array must stay open until the launch has been synchronized.
func ArgTexture(t *Texture) KernelArg {
	return textureKernelArg{t: t}
}

func (a textureKernelArg) appendKernelArg(b *kernelArgBuilder) error {
	if a.t == nil {
		return ErrNilTexture
	}
	held := b.holdsLock(&a.t.opMu)
	if !held {
		a.t.opMu.RLock()
	}
	if a.t.closed {
		if !held {
			a.t.opMu.RUnlock()
		}
		return ErrTextureClosed
	}
	if b.ctx != nil && a.t.ctx != b.ctx {
		if !held {
			a.t.opMu.RUnlock()
		}
		return ErrContextMismatch
	}
	handle := a.t.handle
	if b.snapshot {
		if !held {
			a.t.opMu.RUnlock()
		}
	} else if !held {
		b.addLock(&a.t.opMu)
	}
	argpack.Add(b.packed, handle)
	return nil
}
