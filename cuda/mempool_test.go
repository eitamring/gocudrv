package cuda

import (
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func withPoolFuncs(ctx *Context) {
	ctx.driver.CuDeviceGetDefaultMemPool = func(pool *cudasys.CUmemoryPool, _ cudasys.CUdevice) cudasys.CUresult {
		*pool = 0x9001
		return cudasys.CUDA_SUCCESS
	}
	var stored uint64
	ctx.driver.CuMemPoolSetAttribute = func(_ cudasys.CUmemoryPool, _ int32, value unsafe.Pointer) cudasys.CUresult {
		stored = *(*uint64)(value)
		return cudasys.CUDA_SUCCESS
	}
	ctx.driver.CuMemPoolGetAttribute = func(_ cudasys.CUmemoryPool, _ int32, value unsafe.Pointer) cudasys.CUresult {
		*(*uint64)(value) = stored
		return cudasys.CUDA_SUCCESS
	}
}

func TestDefaultMemPoolAttrs(t *testing.T) {
	var calls memCalls
	ctx, _, _, _ := newAsyncCopyFixture(t, &calls)
	withPoolFuncs(ctx)

	pool, err := ctx.DefaultMemPool()
	if err != nil {
		t.Fatalf("DefaultMemPool: %v", err)
	}
	if pool.raw != 0x9001 {
		t.Errorf("pool raw = %#x, want 0x9001", pool.raw)
	}
	if err := pool.SetReleaseThreshold(8192); err != nil {
		t.Fatalf("SetReleaseThreshold: %v", err)
	}
	got, err := pool.ReleaseThreshold()
	if err != nil {
		t.Fatalf("ReleaseThreshold: %v", err)
	}
	if got != 8192 {
		t.Errorf("threshold = %d, want 8192", got)
	}
	if _, err := pool.UsedMemCurrent(); err != nil {
		t.Errorf("UsedMemCurrent: %v", err)
	}
}

func TestAllocFromPool(t *testing.T) {
	var calls memCalls
	ctx, stream, _, _ := newAsyncCopyFixture(t, &calls)
	withPoolFuncs(ctx)
	var allocBytes atomic.Uint64
	var allocPool atomic.Uintptr
	ctx.driver.CuMemAllocFromPoolAsync = func(ptr *cudasys.CUdeviceptr, b uint64, pool cudasys.CUmemoryPool, _ cudasys.CUstream) cudasys.CUresult {
		allocBytes.Store(b)
		allocPool.Store(uintptr(pool))
		*ptr = 0xDEAD
		return cudasys.CUDA_SUCCESS
	}

	pool, err := ctx.DefaultMemPool()
	if err != nil {
		t.Fatalf("DefaultMemPool: %v", err)
	}
	buf, err := AllocFromPool[float32](pool, stream, 16)
	if err != nil {
		t.Fatalf("AllocFromPool: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	if allocBytes.Load() != 16*4 {
		t.Errorf("alloc bytes = %d, want 64", allocBytes.Load())
	}
	if allocPool.Load() != 0x9001 {
		t.Errorf("alloc pool = %#x, want 0x9001", allocPool.Load())
	}
	if buf.Len() != 16 {
		t.Errorf("Len = %d, want 16", buf.Len())
	}
	if err := buf.FreeAsync(stream); err != nil {
		t.Errorf("FreeAsync: %v", err)
	}
}

func TestAllocFromPoolRejects(t *testing.T) {
	var calls memCalls
	ctx, stream, _, _ := newAsyncCopyFixture(t, &calls)
	withPoolFuncs(ctx)
	pool, err := ctx.DefaultMemPool()
	if err != nil {
		t.Fatalf("DefaultMemPool: %v", err)
	}
	if _, err := AllocFromPool[float32](nil, stream, 4); !errors.Is(err, ErrNilMemPool) {
		t.Errorf("nil pool = %v, want ErrNilMemPool", err)
	}
	if _, err := AllocFromPool[float32](pool, nil, 4); !errors.Is(err, ErrNilStream) {
		t.Errorf("nil stream = %v, want ErrNilStream", err)
	}
	if _, err := AllocFromPool[float32](pool, stream, 0); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero n = %v, want ErrInvalidLength", err)
	}
}

func TestMemPoolSymbolUnavailable(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0x1000)) // no pool funcs bound
	if _, err := ctx.DefaultMemPool(); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("err = %v, want ErrSymbolUnavailable", err)
	}
}

func TestNilMemPool(t *testing.T) {
	var p *MemoryPool
	if _, err := p.ReleaseThreshold(); !errors.Is(err, ErrNilMemPool) {
		t.Errorf("nil ReleaseThreshold = %v, want ErrNilMemPool", err)
	}
	if err := p.SetReleaseThreshold(1); !errors.Is(err, ErrNilMemPool) {
		t.Errorf("nil SetReleaseThreshold = %v, want ErrNilMemPool", err)
	}
}
