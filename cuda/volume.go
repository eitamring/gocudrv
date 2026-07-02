package cuda

import (
	"context"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// Volume is a 3D device allocation whose rows are padded to a driver-chosen
// pitch for coalesced access. Width, Height, and Depth are in elements; Pitch is
// the row stride in bytes and is at least Width*sizeof(T). It is laid out as
// Depth slices of Height rows of Pitch bytes and copied with cuMemcpy3D. Close
// it before the owning Context.
type Volume[T Supported] struct {
	ctx    *Context
	ptr    cudasys.CUdeviceptr
	pitch  uint64
	width  int
	height int
	depth  int
	opMu   sync.RWMutex
	closed bool
}

// AllocVolume allocates a width-by-height-by-depth region (in elements) with
// cuMemAllocPitch over Height*Depth padded rows. It rejects a nil context,
// non-positive dimensions, and byte-size overflow, and returns
// ErrSymbolUnavailable on a driver without cuMemAllocPitch.
func AllocVolume[T Supported](ctx *Context, width, height, depth int) (*Volume[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if width <= 0 || height <= 0 || depth <= 0 {
		return nil, ErrInvalidLength
	}
	es := elemSize[T]()
	if uint64(width) > math.MaxUint64/es {
		return nil, ErrInvalidLength
	}
	if width > math.MaxInt/height || width*height > math.MaxInt/depth {
		return nil, ErrInvalidLength
	}
	widthBytes := uint64(width) * es
	rows := uint64(height) * uint64(depth)
	if widthBytes > math.MaxUint64/rows {
		return nil, ErrInvalidLength
	}
	esb := uint32(4)
	if es == 8 {
		esb = 8
	}

	var ptr cudasys.CUdeviceptr
	var pitch uint64
	err := ctx.do(context.Background(), func() error {
		p, pp, e := cudaresult.MemAllocPitch(ctx.driver, widthBytes, rows, esb)
		if e != nil {
			return e
		}
		ptr, pitch = p, pp
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Volume[T]{ctx: ctx, ptr: ptr, pitch: pitch, width: width, height: height, depth: depth}, nil
}

// Width is the row length in elements. It is 0 for a nil receiver.
func (v *Volume[T]) Width() int {
	if v == nil {
		return 0
	}
	return v.width
}

// Height is the number of rows per slice. It is 0 for a nil receiver.
func (v *Volume[T]) Height() int {
	if v == nil {
		return 0
	}
	return v.height
}

// Depth is the number of slices. It is 0 for a nil receiver.
func (v *Volume[T]) Depth() int {
	if v == nil {
		return 0
	}
	return v.depth
}

// Pitch is the row stride in bytes chosen by the driver. It is 0 for a nil
// receiver.
func (v *Volume[T]) Pitch() uint64 {
	if v == nil {
		return 0
	}
	return v.pitch
}

// DevicePtr is the device pointer at the start of the allocation, a raw snapshot
// valid only while the volume is open. It is 0 for a nil receiver.
func (v *Volume[T]) DevicePtr() cudasys.CUdeviceptr {
	if v == nil {
		return 0
	}
	return v.ptr
}

func (v *Volume[T]) elements() (int, error) {
	if v.width > math.MaxInt/v.height {
		return 0, ErrInvalidLength
	}
	plane := v.width * v.height
	if plane > math.MaxInt/v.depth {
		return 0, ErrInvalidLength
	}
	return plane * v.depth, nil
}

func (v *Volume[T]) hostDesc(hostToDevice bool, host unsafe.Pointer) cudasys.Memcpy3D {
	widthBytes := uint64(v.width) * elemSize[T]()
	dev := cudasys.Memcpy3D{
		WidthInBytes: widthBytes,
		Height:       uint64(v.height),
		Depth:        uint64(v.depth),
	}
	if hostToDevice {
		dev.SrcMemoryType, dev.SrcHost, dev.SrcPitch, dev.SrcHeight = cudasys.MemoryTypeHost, host, widthBytes, uint64(v.height)
		dev.DstMemoryType, dev.DstDevice, dev.DstPitch, dev.DstHeight = cudasys.MemoryTypeDevice, v.ptr, v.pitch, uint64(v.height)
	} else {
		dev.SrcMemoryType, dev.SrcDevice, dev.SrcPitch, dev.SrcHeight = cudasys.MemoryTypeDevice, v.ptr, v.pitch, uint64(v.height)
		dev.DstMemoryType, dev.DstHost, dev.DstPitch, dev.DstHeight = cudasys.MemoryTypeHost, host, widthBytes, uint64(v.height)
	}
	return dev
}

// CopyFrom copies a packed host slice of Width*Height*Depth elements into the
// volume, padding each row to the pitch. len(src) must equal Width*Height*Depth.
func (v *Volume[T]) CopyFrom(ctx context.Context, src []T) error {
	if v == nil {
		return ErrNilBuffer
	}
	v.opMu.RLock()
	defer v.opMu.RUnlock()
	if v.closed {
		return ErrBufferClosed
	}
	n, err := v.elements()
	if err != nil {
		return err
	}
	if len(src) != n {
		return ErrLengthMismatch
	}
	desc := v.hostDesc(true, unsafe.Pointer(&src[0]))
	e := v.ctx.memcpy3D(ctx, &desc)
	runtime.KeepAlive(src)
	return e
}

// CopyTo copies the volume into a packed host slice of Width*Height*Depth
// elements, dropping the row padding. len(dst) must equal Width*Height*Depth.
func (v *Volume[T]) CopyTo(ctx context.Context, dst []T) error {
	if v == nil {
		return ErrNilBuffer
	}
	v.opMu.RLock()
	defer v.opMu.RUnlock()
	if v.closed {
		return ErrBufferClosed
	}
	n, err := v.elements()
	if err != nil {
		return err
	}
	if len(dst) != n {
		return ErrLengthMismatch
	}
	desc := v.hostDesc(false, unsafe.Pointer(&dst[0]))
	e := v.ctx.memcpy3D(ctx, &desc)
	runtime.KeepAlive(dst)
	return e
}

// Close frees the allocation with cuMemFree. Idempotent; a failed free leaves
// the volume open to retry.
func (v *Volume[T]) Close() error {
	if v == nil {
		return ErrNilBuffer
	}
	v.opMu.Lock()
	defer v.opMu.Unlock()
	if v.closed {
		return nil
	}
	if err := v.ctx.doWait(context.Background(), func() error {
		return cudaresult.MemFree(v.ctx.driver, v.ptr)
	}); err != nil {
		return err
	}
	v.closed = true
	return nil
}
