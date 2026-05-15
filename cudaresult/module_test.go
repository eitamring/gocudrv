package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestModuleLoadData(t *testing.T) {
	image := []byte{'P', 'T', 'X', 0}

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUmodule
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuModuleLoadData: func(m *cudasys.CUmodule, p *byte) cudasys.CUresult {
				got := unsafe.Slice(p, len(image))
				for i := range got {
					if got[i] != image[i] {
						t.Errorf("byte[%d] = %d, want %d", i, got[i], image[i])
					}
				}
				*m = 0xBEEF
				return cudasys.CUDA_SUCCESS
			}},
			0xBEEF,
			nil,
		},
		{
			"invalid ptx",
			&cudasys.Driver{CuModuleLoadData: func(*cudasys.CUmodule, *byte) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_PTX
			}},
			0,
			ErrInvalidPTX,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ModuleLoadData(tc.driver, &image[0])
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
				t.Errorf("module = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestModuleUnload(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuModuleUnload: func(m cudasys.CUmodule) cudasys.CUresult {
				if m != 0xBEEF {
					t.Errorf("module = %#x, want 0xBEEF", m)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuModuleUnload: func(cudasys.CUmodule) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ModuleUnload(tc.driver, 0xBEEF)
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

func TestModuleGetFunction(t *testing.T) {
	name := []byte{'v', 'a', 'd', 'd', 0}

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUfunction
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuModuleGetFunction: func(f *cudasys.CUfunction, m cudasys.CUmodule, n *byte) cudasys.CUresult {
				if m != 0xBEEF {
					t.Errorf("module = %#x, want 0xBEEF", m)
				}
				var i int
				for {
					b := *(*byte)(unsafe.Add(unsafe.Pointer(n), i))
					if b == 0 {
						break
					}
					if i >= len(name)-1 {
						t.Fatalf("name not null-terminated within %d bytes", len(name)-1)
					}
					if b != name[i] {
						t.Errorf("name byte[%d] = %d, want %d", i, b, name[i])
					}
					i++
				}
				if i != len(name)-1 {
					t.Errorf("name length = %d, want %d", i, len(name)-1)
				}
				*f = 0xCAFE
				return cudasys.CUDA_SUCCESS
			}},
			0xCAFE,
			nil,
		},
		{
			"not found",
			&cudasys.Driver{CuModuleGetFunction: func(*cudasys.CUfunction, cudasys.CUmodule, *byte) cudasys.CUresult {
				return cudasys.CUDA_ERROR_NOT_FOUND
			}},
			0,
			ErrNotFound,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ModuleGetFunction(tc.driver, 0xBEEF, &name[0])
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
				t.Errorf("function = %#x, want %#x", got, tc.want)
			}
		})
	}
}
