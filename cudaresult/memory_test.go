package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestMemAlloc(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		bytes   uint64
		want    cudasys.CUdeviceptr
		wantErr error
	}{
		{"nil driver", nil, 1024, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 1024, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemAlloc: func(p *cudasys.CUdeviceptr, b uint64) cudasys.CUresult {
				if b != 4096 {
					t.Errorf("bytes = %d, want 4096", b)
				}
				*p = 0xCAFE
				return cudasys.CUDA_SUCCESS
			}},
			4096,
			0xCAFE,
			nil,
		},
		{
			"out of memory",
			&cudasys.Driver{CuMemAlloc: func(*cudasys.CUdeviceptr, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			1024,
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MemAlloc(tc.driver, tc.bytes)
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
				t.Errorf("ptr = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestMemFree(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemFree: func(p cudasys.CUdeviceptr) cudasys.CUresult {
				if p != 0xCAFE {
					t.Errorf("ptr = %#x, want 0xCAFE", p)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemFree: func(cudasys.CUdeviceptr) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemFree(tc.driver, 0xCAFE)
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

func TestMemcpyHtoD(t *testing.T) {
	srcData := []byte{1, 2, 3, 4, 5}
	src := &srcData[0]

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemcpyHtoD: func(dst cudasys.CUdeviceptr, s *byte, b uint64) cudasys.CUresult {
				if dst != 0xCAFE {
					t.Errorf("dst = %#x, want 0xCAFE", dst)
				}
				if b != uint64(len(srcData)) {
					t.Errorf("bytes = %d, want %d", b, len(srcData))
				}
				got := unsafe.Slice(s, b)
				for i := range got {
					if got[i] != srcData[i] {
						t.Errorf("byte[%d] = %d, want %d", i, got[i], srcData[i])
					}
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemcpyHtoD: func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemcpyHtoD(tc.driver, 0xCAFE, src, uint64(len(srcData)))
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

func TestMemAllocHost(t *testing.T) {
	var storage [16]byte

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		bytes   uint64
		wantErr error
		wantNil bool
	}{
		{"nil driver", nil, 16, ErrNotInitialized, true},
		{"nil func", &cudasys.Driver{}, 16, ErrNotInitialized, true},
		{
			"success",
			&cudasys.Driver{CuMemAllocHost: func(pp **byte, b uint64) cudasys.CUresult {
				if b != 16 {
					t.Errorf("bytes = %d, want 16", b)
				}
				*pp = &storage[0]
				return cudasys.CUDA_SUCCESS
			}},
			16,
			nil,
			false,
		},
		{
			"out of memory",
			&cudasys.Driver{CuMemAllocHost: func(**byte, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			16,
			ErrOutOfMemory,
			true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MemAllocHost(tc.driver, tc.bytes)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if tc.wantNil {
				if got != nil {
					t.Errorf("got non-nil pointer, want nil")
				}
				return
			}
			if got != &storage[0] {
				t.Errorf("pointer = %p, want %p", got, &storage[0])
			}
		})
	}
}

func TestMemFreeHost(t *testing.T) {
	var storage [16]byte
	target := &storage[0]

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemFreeHost: func(p *byte) cudasys.CUresult {
				if p != target {
					t.Errorf("ptr = %p, want %p", p, target)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemFreeHost: func(*byte) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemFreeHost(tc.driver, target)
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

func TestMemcpyDtoH(t *testing.T) {
	dstData := make([]byte, 5)
	dst := &dstData[0]

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
		fill    []byte
	}{
		{"nil driver", nil, ErrNotInitialized, nil},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized, nil},
		{
			"success",
			&cudasys.Driver{CuMemcpyDtoH: func(d *byte, src cudasys.CUdeviceptr, b uint64) cudasys.CUresult {
				if src != 0xCAFE {
					t.Errorf("src = %#x, want 0xCAFE", src)
				}
				if b != 5 {
					t.Errorf("bytes = %d, want 5", b)
				}
				slice := unsafe.Slice(d, b)
				copy(slice, []byte{10, 20, 30, 40, 50})
				return cudasys.CUDA_SUCCESS
			}},
			nil,
			[]byte{10, 20, 30, 40, 50},
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemcpyDtoH: func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
			nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for i := range dstData {
				dstData[i] = 0
			}
			err := MemcpyDtoH(tc.driver, dst, 0xCAFE, 5)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				for i, want := range tc.fill {
					if dstData[i] != want {
						t.Errorf("dst[%d] = %d, want %d", i, dstData[i], want)
					}
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}
