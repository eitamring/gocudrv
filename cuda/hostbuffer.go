package cuda

import (
	"context"
	"math"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
)

// HostBuffer is a typed handle to a region of page-locked host memory
// owned by a Context. CUDA can DMA directly to and from this memory
// without going through its internal staging buffer, so transfers are
// faster than copies from pageable Go slices. Pinned host memory is also
// recommended for predictable async-copy overlap and best throughput
// once async copies land in a later PR; pageable host memory is still
// accepted by the async APIs but its behavior is less predictable.
//
// Lifetime rule: a HostBuffer must be closed before its owning Context
// is closed. After Close, any slice previously returned by Slice points
// at freed memory and must not be used. Pair every AllocHost with a
// deferred Close and close every host buffer before its Context.
type HostBuffer[T Supported] struct {
	ctx    *Context
	ptr    *byte
	length int
	bytes  uint64
	opMu   sync.RWMutex
	closed bool
}

// AllocHost allocates n elements of T as page-locked host memory tied to
// ctx. The caller is responsible for closing the returned HostBuffer
// before closing ctx.
func AllocHost[T Supported](ctx *Context, n int) (*HostBuffer[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if n <= 0 {
		return nil, ErrInvalidLength
	}
	var zero T
	elemSize := uint64(unsafe.Sizeof(zero))
	if uint64(n) > math.MaxUint64/elemSize {
		return nil, ErrInvalidLength
	}
	bytes := uint64(n) * elemSize

	var ptr *byte
	err := ctx.doWait(context.Background(), func() error {
		p, e := cudaresult.MemAllocHost(ctx.driver, bytes)
		if e != nil {
			return e
		}
		ptr = p
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &HostBuffer[T]{
		ctx:    ctx,
		ptr:    ptr,
		length: n,
		bytes:  bytes,
	}, nil
}

// Len returns the number of elements the buffer holds.
func (h *HostBuffer[T]) Len() int {
	if h == nil {
		return 0
	}
	return h.length
}

// Bytes returns the total byte size of the buffer.
func (h *HostBuffer[T]) Bytes() uint64 {
	if h == nil {
		return 0
	}
	return h.bytes
}

// Slice returns a []T view backed by the pinned memory. The slice can be
// read and written directly. Returns nil if the buffer is nil or has been
// closed. The slice becomes invalid after Close; do not retain it past
// that point.
func (h *HostBuffer[T]) Slice() []T {
	if h == nil {
		return nil
	}
	h.opMu.RLock()
	defer h.opMu.RUnlock()
	if h.closed {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(h.ptr)), h.length)
}

// Close releases the pinned host memory. Idempotent after a successful
// free; failures leave the buffer open so callers can retry. Returns
// ErrContextClosed if the owning Context was closed first.
func (h *HostBuffer[T]) Close() error {
	if h == nil {
		return ErrNilBuffer
	}
	h.opMu.Lock()
	defer h.opMu.Unlock()
	if h.closed {
		return nil
	}
	if err := h.ctx.doWait(context.Background(), func() error {
		return cudaresult.MemFreeHost(h.ctx.driver, h.ptr)
	}); err != nil {
		return err
	}
	h.closed = true
	return nil
}
