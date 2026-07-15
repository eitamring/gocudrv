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

// PitchedBuffer is a 2D device allocation whose rows are padded to a
// driver-chosen pitch for coalesced access. Width and Height are in elements;
// Pitch is the row stride in bytes and is at least Width*sizeof(T). Close it
// before the owning Context.
type PitchedBuffer[T Supported] struct {
	ctx    *Context
	ptr    cudasys.CUdeviceptr
	pitch  uint64
	width  int
	height int
	opMu   sync.RWMutex
	closed bool
}

// AllocPitched allocates a width-by-height region (in elements) with
// cuMemAllocPitch. It rejects a nil context, non-positive dimensions, and
// byte-size overflow.
func AllocPitched[T Supported](ctx *Context, width, height int) (*PitchedBuffer[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if width <= 0 || height <= 0 {
		return nil, ErrInvalidLength
	}
	es := elemSize[T]()
	if uint64(width) > math.MaxUint64/es {
		return nil, ErrInvalidLength
	}
	widthBytes := uint64(width) * es
	esb := uint32(4)
	if es == 8 {
		esb = 8
	}

	var ptr cudasys.CUdeviceptr
	var pitch uint64
	err := ctx.do(context.Background(), func() error {
		p, pp, e := cudaresult.MemAllocPitch(ctx.driver, widthBytes, uint64(height), esb)
		if e != nil {
			return e
		}
		ptr, pitch = p, pp
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &PitchedBuffer[T]{ctx: ctx, ptr: ptr, pitch: pitch, width: width, height: height}, nil
}

// Width is the row length in elements. It is 0 for a nil receiver.
func (b *PitchedBuffer[T]) Width() int {
	if b == nil {
		return 0
	}
	return b.width
}

// Height is the number of rows. It is 0 for a nil receiver.
func (b *PitchedBuffer[T]) Height() int {
	if b == nil {
		return 0
	}
	return b.height
}

// Pitch is the row stride in bytes chosen by the driver. It is 0 for a nil
// receiver.
func (b *PitchedBuffer[T]) Pitch() uint64 {
	if b == nil {
		return 0
	}
	return b.pitch
}

// DevicePtr is the device pointer at the start of the allocation, a raw
// snapshot valid only while the buffer is open. It is 0 for a nil receiver.
func (b *PitchedBuffer[T]) DevicePtr() cudasys.CUdeviceptr {
	if b == nil {
		return 0
	}
	return b.ptr
}

func (b *PitchedBuffer[T]) elements() (int, error) {
	if b.width > math.MaxInt/b.height {
		return 0, ErrInvalidLength
	}
	return b.width * b.height, nil
}

func lockPitchedPair[T Supported](a, b *PitchedBuffer[T]) func() {
	first, second := a, b
	if uintptr(unsafe.Pointer(first)) > uintptr(unsafe.Pointer(second)) {
		first, second = second, first
	}
	first.opMu.RLock()
	if first == second {
		return first.opMu.RUnlock
	}
	second.opMu.RLock()
	return func() {
		second.opMu.RUnlock()
		first.opMu.RUnlock()
	}
}

// CopyFrom copies a packed host slice of Width*Height elements into the pitched
// buffer, padding each row to the pitch. len(src) must equal Width*Height.
func (b *PitchedBuffer[T]) CopyFrom(ctx context.Context, src []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	n, err := b.elements()
	if err != nil {
		return err
	}
	if len(src) != n {
		return ErrLengthMismatch
	}
	widthBytes := uint64(b.width) * elemSize[T]()
	desc := cudasys.Memcpy2D{
		SrcMemoryType: cudasys.MemoryTypeHost,
		SrcHost:       unsafe.Pointer(&src[0]),
		SrcPitch:      widthBytes,
		DstMemoryType: cudasys.MemoryTypeDevice,
		DstDevice:     b.ptr,
		DstPitch:      b.pitch,
		WidthInBytes:  widthBytes,
		Height:        uint64(b.height),
	}
	e := b.ctx.memcpy2D(ctx, &desc)
	runtime.KeepAlive(src)
	return e
}

// CopyTo copies the pitched buffer into a packed host slice of Width*Height
// elements, dropping the row padding. len(dst) must equal Width*Height.
func (b *PitchedBuffer[T]) CopyTo(ctx context.Context, dst []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	n, err := b.elements()
	if err != nil {
		return err
	}
	if len(dst) != n {
		return ErrLengthMismatch
	}
	widthBytes := uint64(b.width) * elemSize[T]()
	desc := cudasys.Memcpy2D{
		SrcMemoryType: cudasys.MemoryTypeDevice,
		SrcDevice:     b.ptr,
		SrcPitch:      b.pitch,
		DstMemoryType: cudasys.MemoryTypeHost,
		DstHost:       unsafe.Pointer(&dst[0]),
		DstPitch:      widthBytes,
		WidthInBytes:  widthBytes,
		Height:        uint64(b.height),
	}
	e := b.ctx.memcpy2D(ctx, &desc)
	runtime.KeepAlive(dst)
	return e
}

// CopyToDevice copies this buffer into dst, another pitched buffer of equal
// Width and Height in the same context. Pitches may differ.
func (b *PitchedBuffer[T]) CopyToDevice(ctx context.Context, dst *PitchedBuffer[T]) error {
	if b == nil || dst == nil {
		return ErrNilBuffer
	}
	unlock := lockPitchedPair(b, dst)
	defer unlock()
	if b.closed || dst.closed {
		return ErrBufferClosed
	}
	if dst.ctx != b.ctx {
		return ErrContextMismatch
	}
	if dst.width != b.width || dst.height != b.height {
		return ErrLengthMismatch
	}
	widthBytes := uint64(b.width) * elemSize[T]()
	desc := cudasys.Memcpy2D{
		SrcMemoryType: cudasys.MemoryTypeDevice,
		SrcDevice:     b.ptr,
		SrcPitch:      b.pitch,
		DstMemoryType: cudasys.MemoryTypeDevice,
		DstDevice:     dst.ptr,
		DstPitch:      dst.pitch,
		WidthInBytes:  widthBytes,
		Height:        uint64(b.height),
	}
	return b.ctx.memcpy2D(ctx, &desc)
}

// Close frees the pitched allocation with cuMemFree. Idempotent; a failed free
// leaves the buffer open to retry.
func (b *PitchedBuffer[T]) Close() error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.Lock()
	defer b.opMu.Unlock()
	if b.closed {
		return nil
	}
	if err := b.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.MemFree(b.ctx.driver, b.ptr)
	}); err != nil {
		return err
	}
	b.closed = true
	return nil
}
