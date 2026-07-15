package cuda

import (
	"context"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// MemoryType classifies where a pointer's memory lives. It mirrors CUmemorytype.
type MemoryType uint32

const (
	MemoryTypeHost    MemoryType = 1
	MemoryTypeDevice  MemoryType = 2
	MemoryTypeArray   MemoryType = 3
	MemoryTypeUnified MemoryType = 4
)

const pointerAttributeMemoryType int32 = 2 // CU_POINTER_ATTRIBUTE_MEMORY_TYPE

// CanAccessPeer reports whether this device can directly read and write memory
// allocated on peer, the prerequisite for peer-to-peer access and the faster
// path for CopyToPeer. It returns ErrSymbolUnavailable on a driver that does not
// export cuDeviceCanAccessPeer.
func (d *Device) CanAccessPeer(peer *Device) (bool, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return false, ErrNotInitialized
	}
	if d == nil || peer == nil {
		return false, ErrNilDevice
	}
	return cudaresult.DeviceCanAccessPeer(driver, d.handle, peer.handle)
}

// EnablePeerAccess lets this context access memory in peer's context. Both
// contexts must be open and their devices must report CanAccessPeer. It returns
// ErrSymbolUnavailable on a driver that does not export cuCtxEnablePeerAccess.
func (c *Context) EnablePeerAccess(peer *Context) error {
	if c == nil || peer == nil {
		return ErrNilContext
	}
	peer.opMu.RLock()
	defer peer.opMu.RUnlock()
	if peer.closed.Load() {
		return ErrContextClosed
	}
	raw := peer.raw
	return c.do(context.Background(), func() error {
		return cudaresult.CtxEnablePeerAccess(c.driver, raw)
	})
}

// DisablePeerAccess undoes a prior EnablePeerAccess for peer's context.
func (c *Context) DisablePeerAccess(peer *Context) error {
	if c == nil || peer == nil {
		return ErrNilContext
	}
	peer.opMu.RLock()
	defer peer.opMu.RUnlock()
	if peer.closed.Load() {
		return ErrContextClosed
	}
	raw := peer.raw
	return c.do(context.Background(), func() error {
		return cudaresult.CtxDisablePeerAccess(c.driver, raw)
	})
}

// PointerMemoryType reports what kind of memory ptr addresses (host, device,
// array, or unified) via cuPointerGetAttribute. It returns ErrSymbolUnavailable
// on a driver that does not export cuPointerGetAttribute.
func (c *Context) PointerMemoryType(ptr cudasys.CUdeviceptr) (MemoryType, error) {
	if c == nil {
		return 0, ErrNilContext
	}
	var mt uint32
	err := c.do(context.Background(), func() error {
		return cudaresult.PointerGetAttribute(c.driver, unsafe.Pointer(&mt), pointerAttributeMemoryType, ptr)
	})
	if err != nil {
		return 0, err
	}
	return MemoryType(mt), nil
}

// CopyToPeer copies this buffer's contents into dst, a buffer of equal length in
// another Context, with a direct device-to-device transfer via cuMemcpyPeer. It
// blocks until the copy completes. EnablePeerAccess is not required but makes
// the transfer faster when the devices support peer access.
func (b *Buffer[T]) CopyToPeer(ctx context.Context, dst *Buffer[T]) error {
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
	if dst.length != b.length {
		return ErrLengthMismatch
	}
	bytes := b.bytes
	dstPtr, dstCtx := dst.ptr, dst.ctx.raw
	srcPtr, srcCtx := b.ptr, b.ctx.raw
	return b.ctx.doCopyWait(ctx, func() error {
		return cudaresult.MemcpyPeer(b.ctx.driver, dstPtr, dstCtx, srcPtr, srcCtx, bytes)
	})
}
