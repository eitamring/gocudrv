package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestPrimaryCtxRetain(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUcontext
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, dev cudasys.CUdevice) cudasys.CUresult {
				if dev != 7 {
					t.Errorf("dev = %v, want 7", dev)
				}
				*ctx = 0xC0FFEE
				return cudasys.CUDA_SUCCESS
			}},
			0xC0FFEE,
			nil,
		},
		{
			"out of memory",
			&cudasys.Driver{CuDevicePrimaryCtxRetain: func(*cudasys.CUcontext, cudasys.CUdevice) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := PrimaryCtxRetain(tc.driver, 7)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("ctx = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPrimaryCtxRelease(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuDevicePrimaryCtxRelease: func(dev cudasys.CUdevice) cudasys.CUresult {
				if dev != 3 {
					t.Errorf("dev = %v, want 3", dev)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid device",
			&cudasys.Driver{CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_DEVICE
			}},
			ErrInvalidDevice,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := PrimaryCtxRelease(tc.driver, 3)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestCtxSetCurrent(t *testing.T) {
	called := false
	d := &cudasys.Driver{
		CuCtxSetCurrent: func(ctx cudasys.CUcontext) cudasys.CUresult {
			called = true
			if ctx != 0xABCD {
				t.Errorf("ctx = %v, want 0xABCD", ctx)
			}
			return cudasys.CUDA_SUCCESS
		},
	}
	if err := CtxSetCurrent(d, 0xABCD); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !called {
		t.Error("driver function was not called")
	}

	if err := CtxSetCurrent(nil, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver err = %v, want ErrNotInitialized", err)
	}
	if err := CtxSetCurrent(&cudasys.Driver{}, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil func err = %v, want ErrNotInitialized", err)
	}
}

func TestCtxGetCurrent(t *testing.T) {
	d := &cudasys.Driver{
		CuCtxGetCurrent: func(ctx *cudasys.CUcontext) cudasys.CUresult {
			*ctx = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
	}
	got, err := CtxGetCurrent(d)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != 0xBEEF {
		t.Errorf("ctx = %v, want 0xBEEF", got)
	}

	if _, err := CtxGetCurrent(nil); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver err = %v, want ErrNotInitialized", err)
	}
}

func TestCtxSynchronize(t *testing.T) {
	called := false
	d := &cudasys.Driver{
		CuCtxSynchronize: func() cudasys.CUresult {
			called = true
			return cudasys.CUDA_SUCCESS
		},
	}
	if err := CtxSynchronize(d); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !called {
		t.Error("driver function was not called")
	}

	if err := CtxSynchronize(nil); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver err = %v, want ErrNotInitialized", err)
	}
	if err := CtxSynchronize(&cudasys.Driver{}); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil func err = %v, want ErrNotInitialized", err)
	}

	failing := &cudasys.Driver{
		CuCtxSynchronize: func() cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_CONTEXT },
	}
	if err := CtxSynchronize(failing); !errors.Is(err, ErrInvalidContext) {
		t.Errorf("err = %v, want ErrInvalidContext", err)
	}
}

func TestCtxGetStreamPriorityRange(t *testing.T) {
	cases := []struct {
		name         string
		driver       *cudasys.Driver
		wantLeast    int32
		wantGreatest int32
		wantErr      error
	}{
		{"nil driver", nil, 0, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuCtxGetStreamPriorityRange: func(least, greatest *int32) cudasys.CUresult {
				*least = 0
				*greatest = -2
				return cudasys.CUDA_SUCCESS
			}},
			0,
			-2,
			nil,
		},
		{
			"invalid context",
			&cudasys.Driver{CuCtxGetStreamPriorityRange: func(*int32, *int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_CONTEXT
			}},
			0,
			0,
			ErrInvalidContext,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			least, greatest, err := CtxGetStreamPriorityRange(tc.driver)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if least != tc.wantLeast || greatest != tc.wantGreatest {
				t.Errorf("range = (%d, %d), want (%d, %d)", least, greatest, tc.wantLeast, tc.wantGreatest)
			}
		})
	}
}
