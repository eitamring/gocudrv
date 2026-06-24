package cuda

import (
	"context"
	"runtime"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// elemSize is the size in bytes of one element of T.
func elemSize[T Supported]() uint64 {
	var z T
	return uint64(unsafe.Sizeof(z))
}

// offsetPtr is the device pointer for element index off within b.
func (b *Buffer[T]) offsetPtr(off int) cudasys.CUdeviceptr {
	return b.ptr + cudasys.CUdeviceptr(uint64(off)*elemSize[T]())
}

// CopyFromAt copies len(src) elements from the host slice into the buffer
// starting at element dstOffset. It blocks until the copy completes, with the
// same cancellation and host-aliasing semantics as CopyFrom. It returns
// ErrInvalidLength for a negative offset or an empty source, and ErrOutOfRange
// if the destination range does not fit the buffer; both are checked before any
// CUDA call.
func (b *Buffer[T]) CopyFromAt(ctx context.Context, dstOffset int, src []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if dstOffset < 0 || len(src) == 0 {
		return ErrInvalidLength
	}
	if dstOffset > b.length-len(src) {
		return ErrOutOfRange
	}
	srcPtr := (*byte)(unsafe.Pointer(&src[0]))
	dst := b.offsetPtr(dstOffset)
	bytes := uint64(len(src)) * elemSize[T]()
	err := b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyHtoD(b.ctx.driver, dst, srcPtr, bytes)
	})
	runtime.KeepAlive(src)
	return err
}

// CopyToAt copies len(dst) elements from the buffer, starting at element
// srcOffset, into the host slice. It blocks until the copy completes.
// Validation mirrors CopyFromAt.
func (b *Buffer[T]) CopyToAt(ctx context.Context, dst []T, srcOffset int) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if srcOffset < 0 || len(dst) == 0 {
		return ErrInvalidLength
	}
	if srcOffset > b.length-len(dst) {
		return ErrOutOfRange
	}
	dstPtr := (*byte)(unsafe.Pointer(&dst[0]))
	src := b.offsetPtr(srcOffset)
	bytes := uint64(len(dst)) * elemSize[T]()
	err := b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyDtoH(b.ctx.driver, dstPtr, src, bytes)
	})
	runtime.KeepAlive(dst)
	return err
}

// CopyToDeviceAt copies n elements from this buffer starting at srcOffset into
// dst starting at dstOffset, on the same context. It blocks until the copy
// completes. It returns ErrInvalidLength for a non-positive count or a negative
// offset, ErrOutOfRange if either range does not fit its buffer, and
// ErrContextMismatch if dst belongs to a different context; all are checked
// before any CUDA call.
func (b *Buffer[T]) CopyToDeviceAt(ctx context.Context, dstOffset int, dst *Buffer[T], srcOffset, n int) error {
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
	if n <= 0 || srcOffset < 0 || dstOffset < 0 {
		return ErrInvalidLength
	}
	if srcOffset > b.length-n || dstOffset > dst.length-n {
		return ErrOutOfRange
	}
	srcP := b.offsetPtr(srcOffset)
	dstP := dst.offsetPtr(dstOffset)
	bytes := uint64(n) * elemSize[T]()
	return b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyDtoD(b.ctx.driver, dstP, srcP, bytes)
	})
}

// CopyToDeviceAtAsync enqueues the device-to-device offset copy of
// CopyToDeviceAt on stream. It returns after CUDA accepts the work, not after
// the copy finishes, so the caller must not close b, dst, or stream until
// stream.Synchronize confirms the copy is done. Validation matches
// CopyToDeviceAt and runs before any CUDA call.
func (b *Buffer[T]) CopyToDeviceAtAsync(ctx context.Context, stream *Stream, dstOffset int, dst *Buffer[T], srcOffset, n int) error {
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
	if n <= 0 || srcOffset < 0 || dstOffset < 0 {
		return ErrInvalidLength
	}
	if srcOffset > b.length-n || dstOffset > dst.length-n {
		return ErrOutOfRange
	}
	srcP := b.offsetPtr(srcOffset)
	dstP := dst.offsetPtr(dstOffset)
	bytes := uint64(n) * elemSize[T]()
	return b.ctx.doWait(ctx, func() error {
		return cudaresult.MemcpyDtoDAsync(b.ctx.driver, dstP, srcP, bytes, stream.raw)
	})
}
