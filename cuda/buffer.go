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

// AllocAsync enqueues a stream-ordered allocation of n elements of T on the
// device tied to ctx. It returns after CUDA accepts the work, not after the
// allocation is ready, so the memory must not be accessed until stream reaches
// this point (for example after stream.Synchronize or a later op on the same
// stream). Free the returned Buffer with FreeAsync on a stream, or with Close
// once the stream work that uses it has completed. The caller must close the
// buffer before closing ctx.
func AllocAsync[T Supported](ctx *Context, stream *Stream, n int) (*Buffer[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if stream == nil {
		return nil, ErrNilStream
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

	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return nil, ErrStreamClosed
	}
	if stream.ctx != ctx {
		return nil, ErrContextMismatch
	}

	var ptr cudasys.CUdeviceptr
	err := ctx.doWait(context.Background(), func() error {
		p, e := cudaresult.MemAllocAsync(ctx.driver, bytes, stream.raw)
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
	if err := b.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.MemFree(b.ctx.driver, b.ptr)
	}); err != nil {
		return err
	}
	b.closed = true
	return nil
}

// FreeAsync enqueues a stream-ordered free of the buffer's device memory on
// stream. It returns after CUDA accepts the work, not after the memory is
// reclaimed. The memory stays valid for earlier work already queued on the
// same stream, but the caller must not use the buffer afterwards. Idempotent
// after a successful free; failures leave the buffer open so callers can
// retry. Returns ErrContextClosed if the owning Context was closed first and
// ErrContextMismatch if stream belongs to a different Context.
func (b *Buffer[T]) FreeAsync(stream *Stream) error {
	if b == nil {
		return ErrNilBuffer
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.Lock()
	defer b.opMu.Unlock()
	if b.closed {
		return nil
	}
	if stream.ctx != b.ctx {
		return ErrContextMismatch
	}
	if err := b.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.MemFreeAsync(b.ctx.driver, b.ptr, stream.raw)
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
	err := b.ctx.memcpyHtoD(ctx, b.ptr, srcPtr, b.bytes)
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
	err := b.ctx.memcpyDtoH(ctx, dstPtr, b.ptr, b.bytes)
	runtime.KeepAlive(dst)
	return err
}

// CopyFromHost copies a pinned host buffer into the device buffer. It holds
// the host memory's read lock for the duration of the copy so it cannot be
// released while CUDA is reading it.
func (b *Buffer[T]) CopyFromHost(ctx context.Context, src PinnedHost[T]) error {
	if b == nil {
		return ErrNilBuffer
	}
	host, err := pinnedHostRefOf(src)
	if err != nil {
		return err
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if host.ctx != b.ctx {
		return ErrContextMismatch
	}
	if host.length != b.length {
		return ErrLengthMismatch
	}
	err = b.ctx.memcpyHtoD(ctx, b.ptr, host.ptr, b.bytes)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyToHost copies the device buffer into pinned host memory. It holds the
// host memory's read lock for the duration of the copy so it cannot be
// released while CUDA is writing to it.
func (b *Buffer[T]) CopyToHost(ctx context.Context, dst PinnedHost[T]) error {
	if b == nil {
		return ErrNilBuffer
	}
	host, err := pinnedHostRefOf(dst)
	if err != nil {
		return err
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if host.ctx != b.ctx {
		return ErrContextMismatch
	}
	if host.length != b.length {
		return ErrLengthMismatch
	}
	err = b.ctx.memcpyDtoH(ctx, host.ptr, b.ptr, b.bytes)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyFromHostAsync enqueues a copy from pinned host memory into the device
// buffer on stream. It returns after CUDA accepts the work. The pinned memory
// must not be read, mutated, or closed until stream.Synchronize confirms the
// copy is done. The buffer and stream must also remain open until then.
func (b *Buffer[T]) CopyFromHostAsync(ctx context.Context, stream *Stream, src PinnedHost[T]) error {
	if b == nil {
		return ErrNilBuffer
	}
	host, err := pinnedHostRefOf(src)
	if err != nil {
		return err
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != b.ctx || host.ctx != b.ctx {
		return ErrContextMismatch
	}
	if host.length != b.length {
		return ErrLengthMismatch
	}
	err = b.ctx.memcpyHtoDAsync(ctx, b.ptr, host.ptr, b.bytes, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyFromHostRangeAsync enqueues an async copy of n elements starting at
// element srcOffset in src into the buffer starting at element dstOffset. It
// returns after CUDA accepts the work, not after the copy finishes.
//
// The copy is ordered with respect to stream only. If the destination is next
// read on the SAME stream, no further synchronization is needed — CUDA
// stream order provides it for free. Reading the destination from a
// different stream, or from the null (legacy default) stream, requires an
// explicit stream.Synchronize (or an event wait) first; gocudrv's own streams
// are CU_STREAM_NON_BLOCKING and therefore do NOT implicitly order against
// the null stream. src must not be read, mutated, or closed, and b and
// stream must not be closed, until that ordering is established.
//
// Validation matches CopyFromAt and CopyToDeviceAtAsync, and runs before any
// CUDA call: ErrInvalidLength for a non-positive n or a negative offset,
// ErrOutOfRange if either the source or destination range does not fit its
// buffer, and ErrContextMismatch if stream or src belong to a different
// context than b.
func (b *Buffer[T]) CopyFromHostRangeAsync(ctx context.Context, stream *Stream, dstOffset int, src PinnedHost[T], srcOffset, n int) error {
	if b == nil {
		return ErrNilBuffer
	}
	host, err := pinnedHostRefOf(src)
	if err != nil {
		return err
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != b.ctx || host.ctx != b.ctx {
		return ErrContextMismatch
	}
	if n <= 0 || srcOffset < 0 || dstOffset < 0 {
		return ErrInvalidLength
	}
	if srcOffset > host.length-n || dstOffset > b.length-n {
		return ErrOutOfRange
	}
	dst := b.offsetPtr(dstOffset)
	srcPtr := (*byte)(unsafe.Add(unsafe.Pointer(host.ptr), uintptr(srcOffset)*uintptr(elemSize[T]())))
	bytes := uint64(n) * elemSize[T]()
	err = b.ctx.memcpyHtoDAsync(ctx, dst, srcPtr, bytes, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyToHostAsync enqueues a copy from the device buffer into pinned host
// memory on stream. It returns after CUDA accepts the work. The pinned memory
// must not be read, mutated, or closed until stream.Synchronize confirms the
// copy is done. The buffer and stream must also remain open until then.
func (b *Buffer[T]) CopyToHostAsync(ctx context.Context, stream *Stream, dst PinnedHost[T]) error {
	if b == nil {
		return ErrNilBuffer
	}
	host, err := pinnedHostRefOf(dst)
	if err != nil {
		return err
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != b.ctx || host.ctx != b.ctx {
		return ErrContextMismatch
	}
	if host.length != b.length {
		return ErrLengthMismatch
	}
	err = b.ctx.memcpyDtoHAsync(ctx, host.ptr, b.ptr, b.bytes, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// Zero sets every byte of the buffer to zero. Blocks until the memset
// completes. Cancellation semantics match CopyFrom.
func (b *Buffer[T]) Zero(ctx context.Context) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	return b.ctx.memset(ctx, b.ptr, 0, b.bytes, 1)
}

// ZeroAsync enqueues a memset that clears the buffer on stream. It returns
// after CUDA accepts the work, not after the memset finishes. The caller must
// not close b or stream until stream.Synchronize confirms the memset is done.
func (b *Buffer[T]) ZeroAsync(ctx context.Context, stream *Stream) error {
	if b == nil {
		return ErrNilBuffer
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if stream.ctx != b.ctx {
		return ErrContextMismatch
	}
	return b.ctx.memsetAsync(ctx, b.ptr, 0, b.bytes, 1, stream.raw)
}

// Fill sets every element of the buffer to v using the device memset
// primitive whose width matches the element size, so it needs no host
// allocation or copy. Blocks until the memset completes. Cancellation
// semantics match CopyFrom.
//
// The CUDA driver has no 64-bit memset, so Fill returns ErrUnsupportedFillType
// for 8-byte element types (int64, uint64, float64); fill those with a kernel.
func (b *Buffer[T]) Fill(ctx context.Context, v T) error {
	if b == nil {
		return ErrNilBuffer
	}
	if unsafe.Sizeof(v) == 8 {
		return ErrUnsupportedFillType
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	return b.ctx.memset(ctx, b.ptr, fillBits(v), uint64(b.length), unsafe.Sizeof(v))
}

// fillBits returns the bytes of v widened to a uint32, the value the memset
// primitives take. Reading v here keeps it off the heap: the pointer does not
// escape into the launch closure.
func fillBits[T Supported](v T) uint32 {
	switch unsafe.Sizeof(v) {
	case 1:
		return uint32(*(*uint8)(unsafe.Pointer(&v)))
	case 2:
		return uint32(*(*uint16)(unsafe.Pointer(&v)))
	default:
		return *(*uint32)(unsafe.Pointer(&v))
	}
}

// FillAsync enqueues a memset that sets every element of the buffer to v on
// stream. It returns after CUDA accepts the work, not after the memset
// finishes. The caller must not close b or stream until stream.Synchronize
// confirms the memset is done. Like Fill, it returns ErrUnsupportedFillType for
// 8-byte element types.
func (b *Buffer[T]) FillAsync(ctx context.Context, stream *Stream, v T) error {
	if b == nil {
		return ErrNilBuffer
	}
	if stream == nil {
		return ErrNilStream
	}
	if unsafe.Sizeof(v) == 8 {
		return ErrUnsupportedFillType
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if stream.ctx != b.ctx {
		return ErrContextMismatch
	}
	return b.ctx.memsetAsync(ctx, b.ptr, fillBits(v), uint64(b.length), unsafe.Sizeof(v), stream.raw)
}

// CopyToDevice copies b.Len() elements from this buffer into another device
// buffer in the same context. Blocks until the copy completes. Cancellation
// semantics match CopyFrom.
func (b *Buffer[T]) CopyToDevice(ctx context.Context, dst *Buffer[T]) error {
	if b == nil || dst == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if dst != b {
		dst.opMu.RLock()
		defer dst.opMu.RUnlock()
		if dst.closed {
			return ErrBufferClosed
		}
	}
	if dst.ctx != b.ctx {
		return ErrContextMismatch
	}
	if dst.length != b.length {
		return ErrLengthMismatch
	}
	return b.ctx.memcpyDtoD(ctx, dst.ptr, b.ptr, b.bytes)
}

// CopyToDeviceAsync enqueues a device-to-device copy from this buffer into dst
// on stream. It returns after CUDA accepts the work, not after the copy
// finishes. The caller must not close b, dst, or stream until
// stream.Synchronize confirms the copy is done.
func (b *Buffer[T]) CopyToDeviceAsync(ctx context.Context, stream *Stream, dst *Buffer[T]) error {
	if b == nil || dst == nil {
		return ErrNilBuffer
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if dst != b {
		dst.opMu.RLock()
		defer dst.opMu.RUnlock()
		if dst.closed {
			return ErrBufferClosed
		}
	}
	if stream.ctx != b.ctx || dst.ctx != b.ctx {
		return ErrContextMismatch
	}
	if dst.length != b.length {
		return ErrLengthMismatch
	}
	return b.ctx.memcpyDtoDAsync(ctx, dst.ptr, b.ptr, b.bytes, stream.raw)
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
