package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestDeviceCanAccessPeer(t *testing.T) {
	if _, err := DeviceCanAccessPeer(nil, 0, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := DeviceCanAccessPeer(&cudasys.Driver{}, 0, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	for _, tc := range []struct {
		raw  int32
		want bool
	}{{1, true}, {0, false}} {
		var gotDev, gotPeer cudasys.CUdevice
		d := &cudasys.Driver{CuDeviceCanAccessPeer: func(can *int32, dev, peer cudasys.CUdevice) cudasys.CUresult {
			gotDev, gotPeer = dev, peer
			*can = tc.raw
			return cudasys.CUDA_SUCCESS
		}}
		can, err := DeviceCanAccessPeer(d, 2, 3)
		if err != nil {
			t.Fatalf("DeviceCanAccessPeer: %v", err)
		}
		if can != tc.want || gotDev != 2 || gotPeer != 3 {
			t.Errorf("can=%v dev=%d peer=%d, want %v, 2, 3", can, gotDev, gotPeer, tc.want)
		}
	}

	dErr := &cudasys.Driver{CuDeviceCanAccessPeer: func(*int32, cudasys.CUdevice, cudasys.CUdevice) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_DEVICE
	}}
	if _, err := DeviceCanAccessPeer(dErr, 0, 1); !errors.Is(err, ErrInvalidDevice) {
		t.Errorf("err = %v, want ErrInvalidDevice", err)
	}
}

func TestCtxEnableDisablePeerAccess(t *testing.T) {
	if err := CtxEnablePeerAccess(nil, 0x1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("enable nil driver = %v, want ErrNotInitialized", err)
	}
	if err := CtxEnablePeerAccess(&cudasys.Driver{}, 0x1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("enable nil func = %v, want ErrSymbolUnavailable", err)
	}
	if err := CtxDisablePeerAccess(nil, 0x1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("disable nil driver = %v, want ErrNotInitialized", err)
	}
	if err := CtxDisablePeerAccess(&cudasys.Driver{}, 0x1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("disable nil func = %v, want ErrSymbolUnavailable", err)
	}

	var enabledCtx cudasys.CUcontext
	var gotFlags uint32
	d := &cudasys.Driver{
		CuCtxEnablePeerAccess: func(peer cudasys.CUcontext, flags uint32) cudasys.CUresult {
			enabledCtx, gotFlags = peer, flags
			return cudasys.CUDA_SUCCESS
		},
		CuCtxDisablePeerAccess: func(peer cudasys.CUcontext) cudasys.CUresult {
			if peer != 0xC0DE {
				t.Errorf("disable peer = %#x, want 0xC0DE", peer)
			}
			return cudasys.CUDA_SUCCESS
		},
	}
	if err := CtxEnablePeerAccess(d, 0xC0DE); err != nil {
		t.Fatalf("enable: %v", err)
	}
	if enabledCtx != 0xC0DE || gotFlags != 0 {
		t.Errorf("enable ctx=%#x flags=%d, want 0xC0DE, 0", enabledCtx, gotFlags)
	}
	if err := CtxDisablePeerAccess(d, 0xC0DE); err != nil {
		t.Fatalf("disable: %v", err)
	}
}

func TestMemcpyPeer(t *testing.T) {
	if err := MemcpyPeer(nil, 0, 0, 0, 0, 16); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := MemcpyPeer(&cudasys.Driver{}, 0, 0, 0, 0, 16); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var got struct {
		dst, src       cudasys.CUdeviceptr
		dstCtx, srcCtx cudasys.CUcontext
		n              uint64
	}
	d := &cudasys.Driver{CuMemcpyPeer: func(dst cudasys.CUdeviceptr, dstCtx cudasys.CUcontext, src cudasys.CUdeviceptr, srcCtx cudasys.CUcontext, n uint64) cudasys.CUresult {
		got.dst, got.dstCtx, got.src, got.srcCtx, got.n = dst, dstCtx, src, srcCtx, n
		return cudasys.CUDA_SUCCESS
	}}
	if err := MemcpyPeer(d, 0xD, 0xDC, 0x5, 0x5C, 256); err != nil {
		t.Fatalf("MemcpyPeer: %v", err)
	}
	if got.dst != 0xD || got.dstCtx != 0xDC || got.src != 0x5 || got.srcCtx != 0x5C || got.n != 256 {
		t.Errorf("got %+v, want dst=0xD dstCtx=0xDC src=0x5 srcCtx=0x5C n=256", got)
	}
}

func TestPointerGetAttribute(t *testing.T) {
	if err := PointerGetAttribute(nil, nil, 2, 0x1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := PointerGetAttribute(&cudasys.Driver{}, nil, 2, 0x1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var gotAttr int32
	var gotPtr cudasys.CUdeviceptr
	d := &cudasys.Driver{CuPointerGetAttribute: func(data unsafe.Pointer, attribute int32, ptr cudasys.CUdeviceptr) cudasys.CUresult {
		gotAttr, gotPtr = attribute, ptr
		*(*uint32)(data) = 2 // CU_MEMORYTYPE_DEVICE
		return cudasys.CUDA_SUCCESS
	}}
	var mt uint32
	if err := PointerGetAttribute(d, unsafe.Pointer(&mt), 2, 0xDEAD); err != nil {
		t.Fatalf("PointerGetAttribute: %v", err)
	}
	if mt != 2 || gotAttr != 2 || gotPtr != 0xDEAD {
		t.Errorf("mt=%d attr=%d ptr=%#x, want 2, 2, 0xDEAD", mt, gotAttr, gotPtr)
	}
}
