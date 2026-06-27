package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestMemAllocManaged(t *testing.T) {
	if _, err := MemAllocManaged(nil, 64, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := MemAllocManaged(&cudasys.Driver{}, 64, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var got struct {
		n     uint64
		flags uint32
	}
	storage := make([]byte, 64)
	d := &cudasys.Driver{CuMemAllocManaged: func(pp **byte, n uint64, flags uint32) cudasys.CUresult {
		got.n, got.flags = n, flags
		*pp = &storage[0]
		return cudasys.CUDA_SUCCESS
	}}
	p, err := MemAllocManaged(d, 64, 1)
	if err != nil {
		t.Fatalf("MemAllocManaged: %v", err)
	}
	if p != &storage[0] || got.n != 64 || got.flags != 1 {
		t.Errorf("got p=%p n=%d flags=%d, want &storage[0], 64, 1", p, got.n, got.flags)
	}

	dErr := &cudasys.Driver{CuMemAllocManaged: func(**byte, uint64, uint32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_OUT_OF_MEMORY
	}}
	if _, err := MemAllocManaged(dErr, 64, 1); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestMemPrefetchAsync(t *testing.T) {
	if err := MemPrefetchAsync(nil, 0, 64, 0, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := MemPrefetchAsync(&cudasys.Driver{}, 0, 64, 0, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var got struct {
		ptr    cudasys.CUdeviceptr
		count  uint64
		dst    cudasys.CUdevice
		stream cudasys.CUstream
	}
	d := &cudasys.Driver{CuMemPrefetchAsync: func(ptr cudasys.CUdeviceptr, count uint64, dst cudasys.CUdevice, s cudasys.CUstream) cudasys.CUresult {
		got.ptr, got.count, got.dst, got.stream = ptr, count, dst, s
		return cudasys.CUDA_SUCCESS
	}}
	if err := MemPrefetchAsync(d, 0xDEAD, 256, -1, 0x5151); err != nil {
		t.Fatalf("MemPrefetchAsync: %v", err)
	}
	if got.ptr != 0xDEAD || got.count != 256 || got.dst != -1 || got.stream != 0x5151 {
		t.Errorf("got %+v, want ptr=0xDEAD count=256 dst=-1 stream=0x5151", got)
	}
}

func TestMemAdvise(t *testing.T) {
	if err := MemAdvise(nil, 0, 64, 1, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := MemAdvise(&cudasys.Driver{}, 0, 64, 1, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var got struct {
		ptr    cudasys.CUdeviceptr
		count  uint64
		advice int32
		dev    cudasys.CUdevice
	}
	d := &cudasys.Driver{CuMemAdvise: func(ptr cudasys.CUdeviceptr, count uint64, advice int32, dev cudasys.CUdevice) cudasys.CUresult {
		got.ptr, got.count, got.advice, got.dev = ptr, count, advice, dev
		return cudasys.CUDA_SUCCESS
	}}
	if err := MemAdvise(d, 0xBEEF, 128, 3, 2); err != nil {
		t.Fatalf("MemAdvise: %v", err)
	}
	if got.ptr != 0xBEEF || got.count != 128 || got.advice != 3 || got.dev != 2 {
		t.Errorf("got %+v, want ptr=0xBEEF count=128 advice=3 dev=2", got)
	}
}
