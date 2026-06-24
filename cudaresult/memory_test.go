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

func TestMemAllocAsync(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		bytes   uint64
		want    cudasys.CUdeviceptr
		wantErr error
	}{
		{"nil driver", nil, 1024, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 1024, 0, ErrSymbolUnavailable},
		{
			"success",
			&cudasys.Driver{CuMemAllocAsync: func(p *cudasys.CUdeviceptr, b uint64, stream cudasys.CUstream) cudasys.CUresult {
				if b != 4096 {
					t.Errorf("bytes = %d, want 4096", b)
				}
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
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
			&cudasys.Driver{CuMemAllocAsync: func(*cudasys.CUdeviceptr, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			1024,
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MemAllocAsync(tc.driver, tc.bytes, 0x5151)
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

func TestMemFreeAsync(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrSymbolUnavailable},
		{
			"success",
			&cudasys.Driver{CuMemFreeAsync: func(p cudasys.CUdeviceptr, stream cudasys.CUstream) cudasys.CUresult {
				if p != 0xCAFE {
					t.Errorf("ptr = %#x, want 0xCAFE", p)
				}
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemFreeAsync: func(cudasys.CUdeviceptr, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemFreeAsync(tc.driver, 0xCAFE, 0x5151)
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

func TestMemcpyHtoDAsync(t *testing.T) {
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
			&cudasys.Driver{CuMemcpyHtoDAsync: func(dst cudasys.CUdeviceptr, s *byte, b uint64, stream cudasys.CUstream) cudasys.CUresult {
				if dst != 0xCAFE {
					t.Errorf("dst = %#x, want 0xCAFE", dst)
				}
				if b != uint64(len(srcData)) {
					t.Errorf("bytes = %d, want %d", b, len(srcData))
				}
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
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
			&cudasys.Driver{CuMemcpyHtoDAsync: func(cudasys.CUdeviceptr, *byte, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemcpyHtoDAsync(tc.driver, 0xCAFE, src, uint64(len(srcData)), 0x5151)
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

func TestMemcpyDtoHAsync(t *testing.T) {
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
			&cudasys.Driver{CuMemcpyDtoHAsync: func(d *byte, src cudasys.CUdeviceptr, b uint64, stream cudasys.CUstream) cudasys.CUresult {
				if src != 0xCAFE {
					t.Errorf("src = %#x, want 0xCAFE", src)
				}
				if b != 5 {
					t.Errorf("bytes = %d, want 5", b)
				}
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				copy(unsafe.Slice(d, b), []byte{10, 20, 30, 40, 50})
				return cudasys.CUDA_SUCCESS
			}},
			nil,
			[]byte{10, 20, 30, 40, 50},
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemcpyDtoHAsync: func(*byte, cudasys.CUdeviceptr, uint64, cudasys.CUstream) cudasys.CUresult {
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
			err := MemcpyDtoHAsync(tc.driver, dst, 0xCAFE, 5, 0x5151)
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

func TestMemGetInfo(t *testing.T) {
	cases := []struct {
		name      string
		driver    *cudasys.Driver
		wantFree  uint64
		wantTotal uint64
		wantErr   error
	}{
		{"nil driver", nil, 0, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemGetInfo: func(free, total *uint64) cudasys.CUresult {
				*free = 1024
				*total = 4096
				return cudasys.CUDA_SUCCESS
			}},
			1024, 4096, nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemGetInfo: func(*uint64, *uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			0, 0, ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			free, total, err := MemGetInfo(tc.driver)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if free != tc.wantFree || total != tc.wantTotal {
				t.Errorf("got (free=%d total=%d), want (free=%d total=%d)", free, total, tc.wantFree, tc.wantTotal)
			}
		})
	}
}

func TestMemcpyDtoD(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemcpyDtoD: func(dst, src cudasys.CUdeviceptr, bytes uint64) cudasys.CUresult {
				if dst != 0xAAAA || src != 0xBBBB {
					t.Errorf("dst=%#x src=%#x, want 0xAAAA/0xBBBB", dst, src)
				}
				if bytes != 64 {
					t.Errorf("bytes = %d, want 64", bytes)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemcpyDtoD: func(cudasys.CUdeviceptr, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemcpyDtoD(tc.driver, 0xAAAA, 0xBBBB, 64)
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

func TestMemcpyDtoDAsync(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemcpyDtoDAsync: func(dst, src cudasys.CUdeviceptr, bytes uint64, stream cudasys.CUstream) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemcpyDtoDAsync: func(cudasys.CUdeviceptr, cudasys.CUdeviceptr, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemcpyDtoDAsync(tc.driver, 0xAAAA, 0xBBBB, 64, 0x5151)
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

func TestMemsetD8(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemsetD8: func(dst cudasys.CUdeviceptr, value uint8, count uint64) cudasys.CUresult {
				if dst != 0xAAAA || value != 0x7 || count != 32 {
					t.Errorf("got dst=%#x value=%d count=%d", dst, value, count)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemsetD8: func(cudasys.CUdeviceptr, uint8, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemsetD8(tc.driver, 0xAAAA, 0x7, 32)
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

func TestMemsetD16(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemsetD16: func(dst cudasys.CUdeviceptr, value uint16, count uint64) cudasys.CUresult {
				if value != 0xBEEF || count != 16 {
					t.Errorf("got value=%#x count=%d", value, count)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemsetD16: func(cudasys.CUdeviceptr, uint16, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemsetD16(tc.driver, 0xAAAA, 0xBEEF, 16)
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

func TestMemsetD16Async(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemsetD16Async: func(_ cudasys.CUdeviceptr, value uint16, _ uint64, stream cudasys.CUstream) cudasys.CUresult {
				if value != 0xBEEF || stream != 0x5151 {
					t.Errorf("got value=%#x stream=%#x", value, stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemsetD16Async: func(cudasys.CUdeviceptr, uint16, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemsetD16Async(tc.driver, 0xAAAA, 0xBEEF, 16, 0x5151)
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

func TestMemsetD32(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemsetD32: func(dst cudasys.CUdeviceptr, value uint32, count uint64) cudasys.CUresult {
				if value != 0xDEADBEEF || count != 8 {
					t.Errorf("got value=%#x count=%d", value, count)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemsetD32: func(cudasys.CUdeviceptr, uint32, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemsetD32(tc.driver, 0xAAAA, 0xDEADBEEF, 8)
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

func TestMemsetD8Async(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemsetD8Async: func(_ cudasys.CUdeviceptr, _ uint8, _ uint64, stream cudasys.CUstream) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemsetD8Async: func(cudasys.CUdeviceptr, uint8, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemsetD8Async(tc.driver, 0xAAAA, 0x7, 32, 0x5151)
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

func TestMemsetD32Async(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuMemsetD32Async: func(_ cudasys.CUdeviceptr, value uint32, _ uint64, stream cudasys.CUstream) cudasys.CUresult {
				if value != 0xDEADBEEF || stream != 0x5151 {
					t.Errorf("got value=%#x stream=%#x", value, stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuMemsetD32Async: func(cudasys.CUdeviceptr, uint32, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := MemsetD32Async(tc.driver, 0xAAAA, 0xDEADBEEF, 8, 0x5151)
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

func TestMemHostRegister(t *testing.T) {
	var b byte
	if err := MemHostRegister(nil, &b, 16, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := MemHostRegister(&cudasys.Driver{}, &b, 16, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var gotBytes uint64
	var gotFlags uint32
	d := &cudasys.Driver{CuMemHostRegister: func(p *byte, bytes uint64, flags uint32) cudasys.CUresult {
		if p != &b {
			t.Error("pointer mismatch")
		}
		gotBytes, gotFlags = bytes, flags
		return cudasys.CUDA_SUCCESS
	}}
	if err := MemHostRegister(d, &b, 64, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if gotBytes != 64 || gotFlags != 1 {
		t.Errorf("bytes=%d flags=%d, want 64, 1", gotBytes, gotFlags)
	}

	dErr := &cudasys.Driver{CuMemHostRegister: func(*byte, uint64, uint32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}}
	if err := MemHostRegister(dErr, &b, 16, 0); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}

func TestMemHostUnregister(t *testing.T) {
	var b byte
	if err := MemHostUnregister(nil, &b); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := MemHostUnregister(&cudasys.Driver{}, &b); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	d := &cudasys.Driver{CuMemHostUnregister: func(p *byte) cudasys.CUresult {
		if p != &b {
			t.Error("pointer mismatch")
		}
		return cudasys.CUDA_SUCCESS
	}}
	if err := MemHostUnregister(d, &b); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}
