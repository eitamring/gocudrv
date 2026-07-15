package cuda

import (
	"context"
	"runtime"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// CopyFromHostAsync enqueues a packed pinned-host copy into the pitched
// buffer. The host memory must not be read, mutated, or closed until the stream
// is synchronized. The buffer and stream must also remain open until then.
func (b *PitchedBuffer[T]) CopyFromHostAsync(ctx context.Context, stream *Stream, src PinnedHost[T]) error {
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
	n, err := b.elements()
	if err != nil {
		return err
	}
	if host.length != n {
		return ErrLengthMismatch
	}
	widthBytes := uint64(b.width) * elemSize[T]()
	desc := cudasys.Memcpy2D{
		SrcMemoryType: cudasys.MemoryTypeHost,
		SrcHost:       unsafe.Pointer(host.ptr),
		SrcPitch:      widthBytes,
		DstMemoryType: cudasys.MemoryTypeDevice,
		DstDevice:     b.ptr,
		DstPitch:      b.pitch,
		WidthInBytes:  widthBytes,
		Height:        uint64(b.height),
	}
	err = b.ctx.memcpy2DAsync(ctx, &desc, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyToHostAsync enqueues a packed copy from the pitched buffer into pinned
// host memory. The host memory must not be read, mutated, or closed until the
// stream is synchronized. The buffer and stream must also remain open.
func (b *PitchedBuffer[T]) CopyToHostAsync(ctx context.Context, stream *Stream, dst PinnedHost[T]) error {
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
	n, err := b.elements()
	if err != nil {
		return err
	}
	if host.length != n {
		return ErrLengthMismatch
	}
	widthBytes := uint64(b.width) * elemSize[T]()
	desc := cudasys.Memcpy2D{
		SrcMemoryType: cudasys.MemoryTypeDevice,
		SrcDevice:     b.ptr,
		SrcPitch:      b.pitch,
		DstMemoryType: cudasys.MemoryTypeHost,
		DstHost:       unsafe.Pointer(host.ptr),
		DstPitch:      widthBytes,
		WidthInBytes:  widthBytes,
		Height:        uint64(b.height),
	}
	err = b.ctx.memcpy2DAsync(ctx, &desc, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyToDeviceAsync enqueues a copy into an equal-sized pitched buffer. The
// source, destination, and stream must remain open until the stream is
// synchronized.
func (b *PitchedBuffer[T]) CopyToDeviceAsync(ctx context.Context, stream *Stream, dst *PitchedBuffer[T]) error {
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
	unlock := lockPitchedPair(b, dst)
	defer unlock()
	if b.closed || dst.closed {
		return ErrBufferClosed
	}
	if stream.ctx != b.ctx || dst.ctx != b.ctx {
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
	return b.ctx.memcpy2DAsync(ctx, &desc, stream.raw)
}

// CopyFromHostAsync enqueues a packed pinned-host copy into the volume. The
// host memory must not be read, mutated, or closed until the stream is
// synchronized. The volume and stream must also remain open until then.
func (v *Volume[T]) CopyFromHostAsync(ctx context.Context, stream *Stream, src PinnedHost[T]) error {
	if v == nil {
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
	v.opMu.RLock()
	defer v.opMu.RUnlock()
	if v.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != v.ctx || host.ctx != v.ctx {
		return ErrContextMismatch
	}
	n, err := v.elements()
	if err != nil {
		return err
	}
	if host.length != n {
		return ErrLengthMismatch
	}
	desc := v.hostDesc(true, unsafe.Pointer(host.ptr))
	err = v.ctx.memcpy3DAsync(ctx, &desc, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyToHostAsync enqueues a packed volume copy into pinned host memory. The
// host memory must not be read, mutated, or closed until the stream is
// synchronized. The volume and stream must also remain open until then.
func (v *Volume[T]) CopyToHostAsync(ctx context.Context, stream *Stream, dst PinnedHost[T]) error {
	if v == nil {
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
	v.opMu.RLock()
	defer v.opMu.RUnlock()
	if v.closed {
		return ErrBufferClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != v.ctx || host.ctx != v.ctx {
		return ErrContextMismatch
	}
	n, err := v.elements()
	if err != nil {
		return err
	}
	if host.length != n {
		return ErrLengthMismatch
	}
	desc := v.hostDesc(false, unsafe.Pointer(host.ptr))
	err = v.ctx.memcpy3DAsync(ctx, &desc, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyFromHostAsync enqueues a packed pinned-host copy into the array. The host
// memory must not be read, mutated, or closed until the stream is synchronized.
// The array and stream must also remain open until then.
func (a *Array2D[T]) CopyFromHostAsync(ctx context.Context, stream *Stream, src PinnedHost[T]) error {
	if a == nil {
		return ErrNilArray
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
	a.opMu.RLock()
	defer a.opMu.RUnlock()
	if a.closed {
		return ErrArrayClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != a.ctx || host.ctx != a.ctx {
		return ErrContextMismatch
	}
	if host.length != a.width*a.height {
		return ErrLengthMismatch
	}
	desc := a.hostDesc(true, unsafe.Pointer(host.ptr))
	err = a.ctx.memcpy2DAsync(ctx, &desc, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}

// CopyToHostAsync enqueues a packed array copy into pinned host memory. The
// host memory must not be read, mutated, or closed until the stream is
// synchronized. The array and stream must also remain open until then.
func (a *Array2D[T]) CopyToHostAsync(ctx context.Context, stream *Stream, dst PinnedHost[T]) error {
	if a == nil {
		return ErrNilArray
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
	a.opMu.RLock()
	defer a.opMu.RUnlock()
	if a.closed {
		return ErrArrayClosed
	}
	host.lock.RLock()
	defer host.lock.RUnlock()
	if *host.closed {
		return ErrBufferClosed
	}
	if stream.ctx != a.ctx || host.ctx != a.ctx {
		return ErrContextMismatch
	}
	if host.length != a.width*a.height {
		return ErrLengthMismatch
	}
	desc := a.hostDesc(false, unsafe.Pointer(host.ptr))
	err = a.ctx.memcpy2DAsync(ctx, &desc, stream.raw)
	runtime.KeepAlive(host.owner)
	return err
}
