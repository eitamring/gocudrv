package cuda

import (
	"errors"
	"sync/atomic"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func registerFixture(t *testing.T) (*Context, *atomic.Int32, *atomic.Int32, *atomic.Uint64) {
	t.Helper()
	var reg, unreg atomic.Int32
	var lastBytes atomic.Uint64
	drv := fakeMemoryDriver(&memCalls{}, 0x20000)
	drv.CuMemHostRegister = func(_ *byte, bytes uint64, _ uint32) cudasys.CUresult {
		reg.Add(1)
		lastBytes.Store(bytes)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemHostUnregister = func(_ *byte) cudasys.CUresult {
		unreg.Add(1)
		return cudasys.CUDA_SUCCESS
	}
	return newTestContext(t, drv), &reg, &unreg, &lastBytes
}

func TestRegisterHost(t *testing.T) {
	ctx, reg, unreg, lastBytes := registerFixture(t)
	mem := make([]float32, 8)

	r, err := RegisterHost(ctx, mem)
	if err != nil {
		t.Fatalf("RegisterHost: %v", err)
	}
	if reg.Load() != 1 {
		t.Errorf("register calls = %d, want 1", reg.Load())
	}
	if lastBytes.Load() != 8*4 {
		t.Errorf("registered bytes = %d, want 32", lastBytes.Load())
	}
	if r.Len() != 8 || r.Bytes() != 32 {
		t.Errorf("Len/Bytes = %d/%d, want 8/32", r.Len(), r.Bytes())
	}
	if &r.Slice()[0] != &mem[0] {
		t.Error("Slice() should alias the registered memory")
	}

	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if unreg.Load() != 1 {
		t.Errorf("unregister calls = %d, want 1", unreg.Load())
	}
	// Idempotent: a second Close does not unregister again.
	if err := r.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if unreg.Load() != 1 {
		t.Errorf("unregister calls after second Close = %d, want 1", unreg.Load())
	}
}

func TestRegisterHostRejects(t *testing.T) {
	ctx, reg, _, _ := registerFixture(t)
	if _, err := RegisterHost[float32](nil, make([]float32, 4)); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	if _, err := RegisterHost(ctx, []float32{}); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("empty slice = %v, want ErrInvalidLength", err)
	}
	if reg.Load() != 0 {
		t.Errorf("a rejected register reached the driver: %d", reg.Load())
	}
}

func TestRegisterHostUnavailable(t *testing.T) {
	drv := fakeMemoryDriver(&memCalls{}, 0x20000) // no CuMemHostRegister bound
	ctx := newTestContext(t, drv)
	if _, err := RegisterHost(ctx, make([]float32, 4)); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("err = %v, want ErrSymbolUnavailable", err)
	}
}

func TestRegisterHostRetry(t *testing.T) {
	var unreg atomic.Int32
	fail := true
	drv := fakeMemoryDriver(&memCalls{}, 0x20000)
	drv.CuMemHostRegister = func(*byte, uint64, uint32) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuMemHostUnregister = func(*byte) cudasys.CUresult {
		unreg.Add(1)
		if fail {
			return cudasys.CUDA_ERROR_INVALID_VALUE
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, drv)

	r, err := RegisterHost(ctx, make([]float32, 4))
	if err != nil {
		t.Fatalf("RegisterHost: %v", err)
	}
	if err := r.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("first Close = %v, want ErrInvalidValue", err)
	}
	fail = false
	if err := r.Close(); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
	if unreg.Load() != 2 {
		t.Errorf("unregister attempts = %d, want 2 (one failed, one retried)", unreg.Load())
	}
}

func TestRegisteredHostNilReceiver(t *testing.T) {
	var r *RegisteredHost[float32]
	if r.Slice() != nil {
		t.Error("nil Slice should be nil")
	}
	if r.Len() != 0 || r.Bytes() != 0 {
		t.Error("nil Len/Bytes should be 0")
	}
	if err := r.Close(); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil Close = %v, want ErrNilBuffer", err)
	}
}
