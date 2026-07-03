package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestIpcMemHandleWrappers(t *testing.T) {
	if _, err := IpcGetMemHandle(nil, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := IpcGetMemHandle(&cudasys.Driver{}, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	want := cudasys.CUipcMemHandle{Data: [64]byte{1, 2, 3}}
	d := &cudasys.Driver{
		CuIpcGetMemHandle: func(h *cudasys.CUipcMemHandle, dptr cudasys.CUdeviceptr) cudasys.CUresult {
			if dptr != 0xD00D {
				t.Errorf("dptr = %#x, want 0xD00D", dptr)
			}
			*h = want
			return cudasys.CUDA_SUCCESS
		},
		CuIpcOpenMemHandle: func(p *cudasys.CUdeviceptr, h cudasys.CUipcMemHandle, flags uint32) cudasys.CUresult {
			if h != want {
				t.Error("handle not forwarded by value")
			}
			if flags != cudasys.IpcMemLazyEnablePeerAccess {
				t.Errorf("flags = %#x, want lazy peer access", flags)
			}
			*p = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
		CuIpcCloseMemHandle: func(dptr cudasys.CUdeviceptr) cudasys.CUresult {
			if dptr != 0xBEEF {
				t.Errorf("close dptr = %#x, want 0xBEEF", dptr)
			}
			return cudasys.CUDA_SUCCESS
		},
	}
	h, err := IpcGetMemHandle(d, 0xD00D)
	if err != nil || h != want {
		t.Fatalf("IpcGetMemHandle = %v, %v", h.Data[:3], err)
	}
	ptr, err := IpcOpenMemHandle(d, h, cudasys.IpcMemLazyEnablePeerAccess)
	if err != nil || ptr != 0xBEEF {
		t.Fatalf("IpcOpenMemHandle = %#x, %v", ptr, err)
	}
	if err := IpcCloseMemHandle(d, ptr); err != nil {
		t.Fatalf("IpcCloseMemHandle: %v", err)
	}
	dErr := &cudasys.Driver{CuIpcOpenMemHandle: func(*cudasys.CUdeviceptr, cudasys.CUipcMemHandle, uint32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_CONTEXT
	}}
	if _, err := IpcOpenMemHandle(dErr, want, 0); !errors.Is(err, ErrInvalidContext) {
		t.Errorf("same-process open = %v, want ErrInvalidContext", err)
	}
}

func TestIpcEventHandleWrappers(t *testing.T) {
	if _, err := IpcGetEventHandle(nil, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := IpcOpenEventHandle(&cudasys.Driver{}, cudasys.CUipcEventHandle{}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	want := cudasys.CUipcEventHandle{Data: [64]byte{9, 9}}
	d := &cudasys.Driver{
		CuIpcGetEventHandle: func(h *cudasys.CUipcEventHandle, ev cudasys.CUevent) cudasys.CUresult {
			if ev != 0xE0E0 {
				t.Errorf("event = %#x, want 0xE0E0", ev)
			}
			*h = want
			return cudasys.CUDA_SUCCESS
		},
		CuIpcOpenEventHandle: func(ev *cudasys.CUevent, h cudasys.CUipcEventHandle) cudasys.CUresult {
			if h != want {
				t.Error("event handle not forwarded by value")
			}
			*ev = 0xFEED
			return cudasys.CUDA_SUCCESS
		},
	}
	h, err := IpcGetEventHandle(d, 0xE0E0)
	if err != nil || h != want {
		t.Fatalf("IpcGetEventHandle = %v, %v", h.Data[:2], err)
	}
	ev, err := IpcOpenEventHandle(d, h)
	if err != nil || ev != 0xFEED {
		t.Fatalf("IpcOpenEventHandle = %#x, %v", ev, err)
	}
}
