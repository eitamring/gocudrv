package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// peerDriver is a fake driver for the peer tests: two devices, primary contexts
// with distinct handles, and device allocation with distinct pointers. Peer and
// pointer entry points are left nil for each test to set.
func peerDriver() *cudasys.Driver {
	var ctxHandle, ptr atomic.Uint64
	ctxHandle.Store(0x1000)
	ptr.Store(0xD000)
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 2; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, ord int32) cudasys.CUresult {
			*dev = cudasys.CUdevice(ord)
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(c *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*c = cudasys.CUcontext(ctxHandle.Add(1) - 1)
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSetCurrent:           func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemAlloc: func(p *cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			*p = cudasys.CUdeviceptr(ptr.Add(0x100) - 0x100)
			return cudasys.CUDA_SUCCESS
		},
		CuMemFree: func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
	}
}

func TestDeviceCanAccessPeer(t *testing.T) {
	d := peerDriver()
	var gotDev, gotPeer cudasys.CUdevice
	d.CuDeviceCanAccessPeer = func(can *int32, dev, peer cudasys.CUdevice) cudasys.CUresult {
		gotDev, gotPeer = dev, peer
		*can = 1
		return cudasys.CUDA_SUCCESS
	}
	installDriver(t, d)
	dev0, _ := GetDevice(0)
	dev1, _ := GetDevice(1)

	ok, err := dev0.CanAccessPeer(dev1)
	if err != nil {
		t.Fatalf("CanAccessPeer: %v", err)
	}
	if !ok || gotDev != 0 || gotPeer != 1 {
		t.Errorf("ok=%v dev=%d peer=%d, want true, 0, 1", ok, gotDev, gotPeer)
	}

	var nilDev *Device
	if _, err := nilDev.CanAccessPeer(dev1); !errors.Is(err, ErrNilDevice) {
		t.Errorf("nil device = %v, want ErrNilDevice", err)
	}
	if _, err := dev0.CanAccessPeer(nil); !errors.Is(err, ErrNilDevice) {
		t.Errorf("nil peer = %v, want ErrNilDevice", err)
	}
}

func TestDeviceCanAccessPeerUnavailable(t *testing.T) {
	installDriver(t, peerDriver()) // no cuDeviceCanAccessPeer bound
	dev0, _ := GetDevice(0)
	dev1, _ := GetDevice(1)
	if _, err := dev0.CanAccessPeer(dev1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("err = %v, want ErrSymbolUnavailable", err)
	}
}

func TestContextEnableDisablePeerAccess(t *testing.T) {
	d := peerDriver()
	var enabled, disabled cudasys.CUcontext
	var flags uint32
	d.CuCtxEnablePeerAccess = func(peer cudasys.CUcontext, f uint32) cudasys.CUresult {
		enabled, flags = peer, f
		return cudasys.CUDA_SUCCESS
	}
	d.CuCtxDisablePeerAccess = func(peer cudasys.CUcontext) cudasys.CUresult {
		disabled = peer
		return cudasys.CUDA_SUCCESS
	}
	installDriver(t, d)
	dev, _ := GetDevice(0)
	a, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })

	if err := a.EnablePeerAccess(b); err != nil {
		t.Fatalf("EnablePeerAccess: %v", err)
	}
	if enabled != b.raw || flags != 0 {
		t.Errorf("enabled=%#x flags=%d, want %#x, 0", enabled, flags, b.raw)
	}
	if err := a.DisablePeerAccess(b); err != nil {
		t.Fatalf("DisablePeerAccess: %v", err)
	}
	if disabled != b.raw {
		t.Errorf("disabled=%#x, want %#x", disabled, b.raw)
	}

	var nilCtx *Context
	if err := nilCtx.EnablePeerAccess(b); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil receiver = %v, want ErrNilContext", err)
	}
	if err := a.EnablePeerAccess(nil); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil peer = %v, want ErrNilContext", err)
	}

	if err := b.Close(); err != nil {
		t.Fatalf("close b: %v", err)
	}
	if err := a.EnablePeerAccess(b); !errors.Is(err, ErrContextClosed) {
		t.Errorf("closed peer = %v, want ErrContextClosed", err)
	}
}

func TestBufferCopyToPeer(t *testing.T) {
	d := peerDriver()
	var got struct {
		dst, src       cudasys.CUdeviceptr
		dstCtx, srcCtx cudasys.CUcontext
		n              uint64
	}
	d.CuMemcpyPeer = func(dst cudasys.CUdeviceptr, dstCtx cudasys.CUcontext, src cudasys.CUdeviceptr, srcCtx cudasys.CUcontext, n uint64) cudasys.CUresult {
		got.dst, got.dstCtx, got.src, got.srcCtx, got.n = dst, dstCtx, src, srcCtx, n
		return cudasys.CUDA_SUCCESS
	}
	installDriver(t, d)
	dev, _ := GetDevice(0)
	ctxA, _ := dev.Primary()
	t.Cleanup(func() { _ = ctxA.Close() })
	ctxB, _ := dev.Primary()
	t.Cleanup(func() { _ = ctxB.Close() })

	src, _ := Alloc[float32](ctxA, 8)
	t.Cleanup(func() { _ = src.Close() })
	dst, _ := Alloc[float32](ctxB, 8)
	t.Cleanup(func() { _ = dst.Close() })

	if err := src.CopyToPeer(context.Background(), dst); err != nil {
		t.Fatalf("CopyToPeer: %v", err)
	}
	if got.src != src.ptr || got.dst != dst.ptr || got.srcCtx != ctxA.raw || got.dstCtx != ctxB.raw || got.n != 32 {
		t.Errorf("got %+v, want src=%#x dst=%#x srcCtx=%#x dstCtx=%#x n=32", got, src.ptr, dst.ptr, ctxA.raw, ctxB.raw)
	}

	short, _ := Alloc[float32](ctxB, 4)
	t.Cleanup(func() { _ = short.Close() })
	if err := src.CopyToPeer(context.Background(), short); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("length mismatch = %v, want ErrLengthMismatch", err)
	}
	if err := src.CopyToPeer(context.Background(), nil); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil dst = %v, want ErrNilBuffer", err)
	}
	// Self-copy must not deadlock on the buffer lock.
	if err := src.CopyToPeer(context.Background(), src); err != nil {
		t.Fatalf("self copy: %v", err)
	}

	if err := src.Close(); err != nil {
		t.Fatalf("close src: %v", err)
	}
	if err := src.CopyToPeer(context.Background(), dst); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("closed src = %v, want ErrBufferClosed", err)
	}
}

func TestContextPointerMemoryType(t *testing.T) {
	d := peerDriver()
	var gotAttr int32
	d.CuPointerGetAttribute = func(data unsafe.Pointer, attribute int32, _ cudasys.CUdeviceptr) cudasys.CUresult {
		gotAttr = attribute
		*(*uint32)(data) = uint32(MemoryTypeDevice)
		return cudasys.CUDA_SUCCESS
	}
	installDriver(t, d)
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })

	mt, err := ctx.PointerMemoryType(0xDEAD)
	if err != nil {
		t.Fatalf("PointerMemoryType: %v", err)
	}
	if mt != MemoryTypeDevice || gotAttr != pointerAttributeMemoryType {
		t.Errorf("mt=%d attr=%d, want %d, %d", mt, gotAttr, MemoryTypeDevice, pointerAttributeMemoryType)
	}
}

func TestContextPointerMemoryTypeUnavailable(t *testing.T) {
	installDriver(t, peerDriver()) // no cuPointerGetAttribute bound
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })
	if _, err := ctx.PointerMemoryType(0xDEAD); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("err = %v, want ErrSymbolUnavailable", err)
	}
}
