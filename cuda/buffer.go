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

// Supported constrains buffer element types to fixed-size numeric scalars.
// Structs and unsized integer types (`int`, `uint`) are intentionally
// excluded to avoid alignment and ABI hazards.
type Supported interface {
	~int8 | ~uint8 |
		~int16 | ~uint16 |
		~int32 | ~uint32 |
		~int64 | ~uint64 |
		~float32 | ~float64
}

// Buffer is a typed handle to a region of device memory owned by a Context.
//
// Lifetime rule: a Buffer must be closed before its owning Context is
// closed. After the Context is closed, Buffer.Close cannot reach the
// executor and returns ErrContextClosed; the underlying device memory is
// reclaimed when the primary context retain count drops to zero, but the
// wrapper cannot guarantee that ordering. Pair every Alloc with a deferred
// Close and close every buffer before its Context.
type Buffer[T Supported] struct {
	ctx    *Context
	ptr    cudasys.CUdeviceptr
	length int
	bytes  uint64
	opMu   sync.RWMutex
	closed bool
}

// Alloc allocates n elements of T on the device tied to ctx. The caller is
// responsible for closing the returned Buffer before closing ctx.
func Alloc[T Supported](ctx *Context, n int) (*Buffer[T], error) {
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

	var ptr cudasys.CUdeviceptr
	err := ctx.do(context.Background(), func() error {
		p, e := cudaresult.MemAlloc(ctx.driver, bytes)
		if e != nil {
			return e
		}
		ptr = p
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Buffer[T]{
		ctx:    ctx,
		ptr:    ptr,
		length: n,
		bytes:  bytes,
	}, nil
}

// Len returns the number of elements in the buffer.
func (b *Buffer[T]) Len() int {
	if b == nil {
		return 0
	}
	return b.length
}

// Bytes returns the total byte size of the buffer.
func (b *Buffer[T]) Bytes() uint64 {
	if b == nil {
		return 0
	}
	return b.bytes
}

// Close releases the device memory. Idempotent after a successful free;
// failures leave the buffer open so callers can retry. Returns
// ErrContextClosed if the owning Context was closed first.
func (b *Buffer[T]) Close() error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.Lock()
	defer b.opMu.Unlock()
	if b.closed {
		return nil
	}
	if err := b.ctx.doWait(context.Background(), func() error {
		return cudaresult.MemFree(b.ctx.driver, b.ptr)
	}); err != nil {
		return err
	}
	b.closed = true
	return nil
}

// CopyFrom copies len(src) elements from the host slice into the buffer.
// Blocks until the copy completes. If ctx is canceled before submission, the
// copy does not run. Once submitted, CopyFrom waits for completion so src is
// not reused while CUDA is still reading it.
func (b *Buffer[T]) CopyFrom(ctx context.Context, src []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if len(src) == 0 || len(src) != b.length {
		return ErrLengthMismatch
	}
	srcPtr := (*byte)(unsafe.Pointer(&src[0]))
	bytes := b.bytes
	err := b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyHtoD(b.ctx.driver, b.ptr, srcPtr, bytes)
	})
	runtime.KeepAlive(src)
	return err
}

// CopyTo copies b.Len() elements from the buffer into the host slice.
// Blocks until the copy completes. Cancellation semantics match CopyFrom.
func (b *Buffer[T]) CopyTo(ctx context.Context, dst []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if len(dst) == 0 || len(dst) != b.length {
		return ErrLengthMismatch
	}
	dstPtr := (*byte)(unsafe.Pointer(&dst[0]))
	bytes := b.bytes
	err := b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyDtoH(b.ctx.driver, dstPtr, b.ptr, bytes)
	})
	runtime.KeepAlive(dst)
	return err
}

// CopyFromHost copies len(src) elements from a pinned HostBuffer into the
// device buffer. Unlike CopyFrom with a raw []T, this method holds the
// host buffer's read lock for the duration of the copy, so the pinned
// memory cannot be freed by HostBuffer.Close while CUDA is reading it.
// Prefer this method when the source is pinned.
func (b *Buffer[T]) CopyFromHost(ctx context.Context, src *HostBuffer[T]) error {
	if b == nil || src == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	src.opMu.RLock()
	defer src.opMu.RUnlock()
	if src.closed {
		return ErrBufferClosed
	}
	if src.length != b.length {
		return ErrLengthMismatch
	}
	bytes := b.bytes
	return b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyHtoD(b.ctx.driver, b.ptr, src.ptr, bytes)
	})
}

// CopyToHost copies b.Len() elements from the device buffer into a pinned
// HostBuffer. Holds the host buffer's read lock for the duration of the
// copy so HostBuffer.Close cannot free the pinned memory while CUDA is
// writing to it. Prefer this method when the destination is pinned.
func (b *Buffer[T]) CopyToHost(ctx context.Context, dst *HostBuffer[T]) error {
	if b == nil || dst == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	dst.opMu.RLock()
	defer dst.opMu.RUnlock()
	if dst.closed {
		return ErrBufferClosed
	}
	if dst.length != b.length {
		return ErrLengthMismatch
	}
	bytes := b.bytes
	return b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyDtoH(b.ctx.driver, dst.ptr, b.ptr, bytes)
	})
}

// CopyHtoD is a thin wrapper around (*Buffer[T]).CopyFrom kept for the
// CUDA-style naming. Prefer the method form in new code.
func CopyHtoD[T Supported](ctx context.Context, dst *Buffer[T], src []T) error {
	return dst.CopyFrom(ctx, src)
}

// CopyDtoH is a thin wrapper around (*Buffer[T]).CopyTo kept for the
// CUDA-style naming. Prefer the method form in new code.
func CopyDtoH[T Supported](ctx context.Context, dst []T, src *Buffer[T]) error {
	return src.CopyTo(ctx, dst)
}
