package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestDeviceGetDefaultMemPool(t *testing.T) {
	if _, err := DeviceGetDefaultMemPool(nil, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := DeviceGetDefaultMemPool(&cudasys.Driver{}, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	d := &cudasys.Driver{CuDeviceGetDefaultMemPool: func(pool *cudasys.CUmemoryPool, _ cudasys.CUdevice) cudasys.CUresult {
		*pool = 0x9001
		return cudasys.CUDA_SUCCESS
	}}
	pool, err := DeviceGetDefaultMemPool(d, 0)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if pool != 0x9001 {
		t.Errorf("pool = %#x, want 0x9001", pool)
	}
}

func TestMemPoolAttributeU64(t *testing.T) {
	if _, err := MemPoolGetAttributeU64(nil, 0, cudasys.MemPoolAttrReleaseThreshold); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("get nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := MemPoolGetAttributeU64(&cudasys.Driver{}, 0, cudasys.MemPoolAttrReleaseThreshold); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("get nil func = %v, want ErrSymbolUnavailable", err)
	}
	if err := MemPoolSetAttributeU64(&cudasys.Driver{}, 0, cudasys.MemPoolAttrReleaseThreshold, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("set nil func = %v, want ErrSymbolUnavailable", err)
	}

	var stored uint64
	var gotAttr int32
	d := &cudasys.Driver{
		CuMemPoolSetAttribute: func(_ cudasys.CUmemoryPool, attr int32, value unsafe.Pointer) cudasys.CUresult {
			gotAttr = attr
			stored = *(*uint64)(value)
			return cudasys.CUDA_SUCCESS
		},
		CuMemPoolGetAttribute: func(_ cudasys.CUmemoryPool, _ int32, value unsafe.Pointer) cudasys.CUresult {
			*(*uint64)(value) = stored
			return cudasys.CUDA_SUCCESS
		},
	}
	if err := MemPoolSetAttributeU64(d, 0x9001, cudasys.MemPoolAttrReleaseThreshold, 4096); err != nil {
		t.Fatalf("set: %v", err)
	}
	if gotAttr != cudasys.MemPoolAttrReleaseThreshold || stored != 4096 {
		t.Errorf("set attr=%d value=%d, want 4, 4096", gotAttr, stored)
	}
	got, err := MemPoolGetAttributeU64(d, 0x9001, cudasys.MemPoolAttrReleaseThreshold)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if got != 4096 {
		t.Errorf("get = %d, want 4096", got)
	}
}

func TestMemAllocFromPoolAsync(t *testing.T) {
	if _, err := MemAllocFromPoolAsync(nil, 16, 0, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := MemAllocFromPoolAsync(&cudasys.Driver{}, 16, 0, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var gotBytes uint64
	var gotPool cudasys.CUmemoryPool
	var gotStream cudasys.CUstream
	d := &cudasys.Driver{CuMemAllocFromPoolAsync: func(ptr *cudasys.CUdeviceptr, b uint64, pool cudasys.CUmemoryPool, s cudasys.CUstream) cudasys.CUresult {
		gotBytes, gotPool, gotStream = b, pool, s
		*ptr = 0xBEEF
		return cudasys.CUDA_SUCCESS
	}}
	ptr, err := MemAllocFromPoolAsync(d, 256, 0x9001, 0x5151)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if ptr != 0xBEEF || gotBytes != 256 || gotPool != 0x9001 || gotStream != 0x5151 {
		t.Errorf("got ptr=%#x bytes=%d pool=%#x stream=%#x", ptr, gotBytes, gotPool, gotStream)
	}
}
