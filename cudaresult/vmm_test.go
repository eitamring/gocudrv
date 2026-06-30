package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestVMMWrappersNilAndUnavailable(t *testing.T) {
	empty := &cudasys.Driver{}
	cases := []struct {
		name    string
		nilDrv  func() error
		unavail func() error
	}{
		{"granularity",
			func() error { _, e := MemGetAllocationGranularity(nil, nil, 1); return e },
			func() error { _, e := MemGetAllocationGranularity(empty, nil, 1); return e }},
		{"create",
			func() error { _, e := MemCreate(nil, 1, nil); return e },
			func() error { _, e := MemCreate(empty, 1, nil); return e }},
		{"reserve",
			func() error { _, e := MemAddressReserve(nil, 1, 1); return e },
			func() error { _, e := MemAddressReserve(empty, 1, 1); return e }},
		{"map", func() error { return MemMap(nil, 0, 1, 0) }, func() error { return MemMap(empty, 0, 1, 0) }},
		{"setaccess", func() error { return MemSetAccess(nil, 0, 1, nil) }, func() error { return MemSetAccess(empty, 0, 1, nil) }},
		{"unmap", func() error { return MemUnmap(nil, 0, 1) }, func() error { return MemUnmap(empty, 0, 1) }},
		{"addressfree", func() error { return MemAddressFree(nil, 0, 1) }, func() error { return MemAddressFree(empty, 0, 1) }},
		{"release", func() error { return MemRelease(nil, 0) }, func() error { return MemRelease(empty, 0) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if err := c.nilDrv(); !errors.Is(err, ErrNotInitialized) {
				t.Errorf("nil driver = %v, want ErrNotInitialized", err)
			}
			if err := c.unavail(); !errors.Is(err, ErrSymbolUnavailable) {
				t.Errorf("unavailable = %v, want ErrSymbolUnavailable", err)
			}
		})
	}
}

func TestVMMLifecycleWrappers(t *testing.T) {
	var got struct {
		gran          uint64
		createSize    uint64
		reserveSize   uint64
		reserveAlign  uint64
		mapPtr        cudasys.CUdeviceptr
		mapHandle     cudasys.CUmemGenericAllocationHandle
		setCount      uint64
		unmapSize     uint64
		freeSize      uint64
		releaseHandle cudasys.CUmemGenericAllocationHandle
	}
	d := &cudasys.Driver{
		CuMemGetAllocationGranularity: func(g *uint64, _ *cudasys.CUmemAllocationProp, opt uint32) cudasys.CUresult {
			got.gran = 2 << 20
			*g = got.gran
			return cudasys.CUDA_SUCCESS
		},
		CuMemCreate: func(h *cudasys.CUmemGenericAllocationHandle, size uint64, _ *cudasys.CUmemAllocationProp, _ uint64) cudasys.CUresult {
			got.createSize = size
			*h = 0xABCD
			return cudasys.CUDA_SUCCESS
		},
		CuMemAddressReserve: func(p *cudasys.CUdeviceptr, size, align uint64, _ cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			got.reserveSize, got.reserveAlign = size, align
			*p = 0x5A000
			return cudasys.CUDA_SUCCESS
		},
		CuMemMap: func(ptr cudasys.CUdeviceptr, _ uint64, _ uint64, h cudasys.CUmemGenericAllocationHandle, _ uint64) cudasys.CUresult {
			got.mapPtr, got.mapHandle = ptr, h
			return cudasys.CUDA_SUCCESS
		},
		CuMemSetAccess: func(_ cudasys.CUdeviceptr, _ uint64, _ *cudasys.CUmemAccessDesc, count uint64) cudasys.CUresult {
			got.setCount = count
			return cudasys.CUDA_SUCCESS
		},
		CuMemUnmap: func(_ cudasys.CUdeviceptr, size uint64) cudasys.CUresult {
			got.unmapSize = size
			return cudasys.CUDA_SUCCESS
		},
		CuMemAddressFree: func(_ cudasys.CUdeviceptr, size uint64) cudasys.CUresult {
			got.freeSize = size
			return cudasys.CUDA_SUCCESS
		},
		CuMemRelease: func(h cudasys.CUmemGenericAllocationHandle) cudasys.CUresult {
			got.releaseHandle = h
			return cudasys.CUDA_SUCCESS
		},
	}

	prop := &cudasys.CUmemAllocationProp{}
	gran, err := MemGetAllocationGranularity(d, prop, 1)
	if err != nil || gran != 2<<20 {
		t.Fatalf("granularity = %d, %v", gran, err)
	}
	h, err := MemCreate(d, gran, prop)
	if err != nil || h != 0xABCD || got.createSize != gran {
		t.Fatalf("create h=%#x size=%d err=%v", h, got.createSize, err)
	}
	ptr, err := MemAddressReserve(d, gran, gran)
	if err != nil || ptr != 0x5A000 || got.reserveSize != gran || got.reserveAlign != gran {
		t.Fatalf("reserve ptr=%#x err=%v", ptr, err)
	}
	if err := MemMap(d, ptr, gran, h); err != nil || got.mapPtr != ptr || got.mapHandle != h {
		t.Fatalf("map err=%v", err)
	}
	if err := MemSetAccess(d, ptr, gran, &cudasys.CUmemAccessDesc{}); err != nil || got.setCount != 1 {
		t.Fatalf("setaccess count=%d err=%v", got.setCount, err)
	}
	if err := MemUnmap(d, ptr, gran); err != nil || got.unmapSize != gran {
		t.Fatalf("unmap err=%v", err)
	}
	if err := MemAddressFree(d, ptr, gran); err != nil || got.freeSize != gran {
		t.Fatalf("free err=%v", err)
	}
	if err := MemRelease(d, h); err != nil || got.releaseHandle != h {
		t.Fatalf("release err=%v", err)
	}
}
