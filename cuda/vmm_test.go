package cuda

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

type vmmRec struct {
	created, reserved, mapped, setAccess int
	unmapped, freed, released            int
	createSize, mapSize                  uint64
	ptr                                  cudasys.CUdeviceptr
	handle                               cudasys.CUmemGenericAllocationHandle
	accessFlags                          int32
	accessLocType, accessLocID           int32
	hostToDev, devToHost                 uint64
}

// vmmDriver returns a fake driver with the context, memcpy, and a recording VMM
// lifecycle the virtual-memory tests need. Tests may override individual entry
// points to inject failures.
func vmmDriver(rec *vmmRec) *cudasys.Driver {
	d := managedDriver()
	rec.ptr = 0x5A000
	rec.handle = 0xABCD
	const gran uint64 = 2 << 20
	d.CuMemGetAllocationGranularity = func(g *uint64, _ *cudasys.CUmemAllocationProp, _ uint32) cudasys.CUresult {
		*g = gran
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemCreate = func(h *cudasys.CUmemGenericAllocationHandle, size uint64, _ *cudasys.CUmemAllocationProp, _ uint64) cudasys.CUresult {
		rec.created++
		rec.createSize = size
		*h = rec.handle
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemAddressReserve = func(p *cudasys.CUdeviceptr, _, _ uint64, _ cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
		rec.reserved++
		*p = rec.ptr
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemMap = func(_ cudasys.CUdeviceptr, size, _ uint64, _ cudasys.CUmemGenericAllocationHandle, _ uint64) cudasys.CUresult {
		rec.mapped++
		rec.mapSize = size
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemSetAccess = func(_ cudasys.CUdeviceptr, _ uint64, desc *cudasys.CUmemAccessDesc, _ uint64) cudasys.CUresult {
		rec.setAccess++
		rec.accessFlags = desc.Flags
		rec.accessLocType, rec.accessLocID = desc.Location.Type, desc.Location.Id
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemUnmap = func(cudasys.CUdeviceptr, uint64) cudasys.CUresult { rec.unmapped++; return cudasys.CUDA_SUCCESS }
	d.CuMemAddressFree = func(cudasys.CUdeviceptr, uint64) cudasys.CUresult { rec.freed++; return cudasys.CUDA_SUCCESS }
	d.CuMemRelease = func(cudasys.CUmemGenericAllocationHandle) cudasys.CUresult {
		rec.released++
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemcpyHtoD = func(_ cudasys.CUdeviceptr, _ *byte, n uint64) cudasys.CUresult {
		rec.hostToDev = n
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemcpyDtoH = func(_ *byte, _ cudasys.CUdeviceptr, n uint64) cudasys.CUresult {
		rec.devToHost = n
		return cudasys.CUDA_SUCCESS
	}
	return d
}

func TestAllocVirtual(t *testing.T) {
	var rec vmmRec
	ctx := newManagedContext(t, vmmDriver(&rec))

	vb, err := AllocVirtual[float32](ctx, 1024)
	if err != nil {
		t.Fatalf("AllocVirtual: %v", err)
	}
	if rec.created != 1 || rec.reserved != 1 || rec.mapped != 1 || rec.setAccess != 1 {
		t.Errorf("lifecycle counts: %+v", rec)
	}
	if rec.createSize != 2<<20 || rec.mapSize != 2<<20 {
		t.Errorf("size not rounded to granularity: create=%d map=%d, want %d", rec.createSize, rec.mapSize, 2<<20)
	}
	if rec.accessFlags != memAccessReadWrite || rec.accessLocType != memLocTypeDevice || rec.accessLocID != 0 {
		t.Errorf("access desc flags=%d locType=%d locID=%d", rec.accessFlags, rec.accessLocType, rec.accessLocID)
	}
	if vb.Len() != 1024 || vb.Bytes() != 4096 || vb.DevicePtr() != 0x5A000 {
		t.Errorf("len=%d bytes=%d ptr=%#x", vb.Len(), vb.Bytes(), vb.DevicePtr())
	}

	if err := vb.CopyFrom(context.Background(), make([]float32, 1024)); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	if err := vb.CopyTo(context.Background(), make([]float32, 1024)); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	if rec.hostToDev != 4096 || rec.devToHost != 4096 {
		t.Errorf("copy bytes h2d=%d d2h=%d, want 4096 each", rec.hostToDev, rec.devToHost)
	}
	if err := vb.CopyFrom(context.Background(), make([]float32, 8)); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("length mismatch = %v, want ErrLengthMismatch", err)
	}

	if err := vb.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if rec.unmapped != 1 || rec.freed != 1 || rec.released != 1 {
		t.Errorf("teardown counts: unmap=%d free=%d release=%d", rec.unmapped, rec.freed, rec.released)
	}
	if err := vb.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if err := vb.CopyFrom(context.Background(), make([]float32, 1024)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyFrom after close = %v, want ErrBufferClosed", err)
	}
}

func TestAllocVirtualRollback(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	d.CuMemMap = func(cudasys.CUdeviceptr, uint64, uint64, cudasys.CUmemGenericAllocationHandle, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_OUT_OF_MEMORY
	}
	ctx := newManagedContext(t, d)

	if _, err := AllocVirtual[float32](ctx, 1024); !errors.Is(err, ErrOutOfMemory) {
		t.Fatalf("AllocVirtual = %v, want ErrOutOfMemory", err)
	}
	if rec.created != 1 || rec.reserved != 1 || rec.freed != 1 || rec.released != 1 {
		t.Errorf("rollback counts: created=%d reserved=%d freed=%d released=%d", rec.created, rec.reserved, rec.freed, rec.released)
	}
}

func TestAllocVirtualRejects(t *testing.T) {
	if _, err := AllocVirtual[float32](nil, 8); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	var rec vmmRec
	ctx := newManagedContext(t, vmmDriver(&rec))
	if _, err := AllocVirtual[float32](ctx, 0); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero n = %v, want ErrInvalidLength", err)
	}

	bare := newManagedContext(t, managedDriver())
	if _, err := AllocVirtual[float32](bare, 8); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("unavailable = %v, want ErrSymbolUnavailable", err)
	}

	var nilVB *VirtualBuffer[float32]
	if err := nilVB.CopyFrom(context.Background(), nil); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil receiver = %v, want ErrNilBuffer", err)
	}
}

func TestAllocVirtualReserveFailRollback(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	d.CuMemAddressReserve = func(*cudasys.CUdeviceptr, uint64, uint64, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_OUT_OF_MEMORY
	}
	ctx := newManagedContext(t, d)
	if _, err := AllocVirtual[float32](ctx, 1024); !errors.Is(err, ErrOutOfMemory) {
		t.Fatalf("AllocVirtual = %v, want ErrOutOfMemory", err)
	}
	if rec.created != 1 || rec.released != 1 || rec.reserved != 0 || rec.mapped != 0 {
		t.Errorf("created=%d released=%d reserved=%d mapped=%d", rec.created, rec.released, rec.reserved, rec.mapped)
	}
}

func TestAllocVirtualSetAccessFailRollback(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	d.CuMemSetAccess = func(cudasys.CUdeviceptr, uint64, *cudasys.CUmemAccessDesc, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	ctx := newManagedContext(t, d)
	if _, err := AllocVirtual[float32](ctx, 1024); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("AllocVirtual = %v, want ErrInvalidValue", err)
	}
	if rec.unmapped != 1 || rec.freed != 1 || rec.released != 1 {
		t.Errorf("unmap=%d free=%d release=%d, want 1,1,1", rec.unmapped, rec.freed, rec.released)
	}
}

func TestVirtualBufferCloseRetry(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	var unmapFails atomic.Bool
	unmapFails.Store(true)
	d.CuMemUnmap = func(cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		if unmapFails.Load() {
			return cudasys.CUDA_ERROR_INVALID_VALUE
		}
		rec.unmapped++
		return cudasys.CUDA_SUCCESS
	}
	ctx := newManagedContext(t, d)
	vb, err := AllocVirtual[float32](ctx, 1024)
	if err != nil {
		t.Fatalf("AllocVirtual: %v", err)
	}

	if err := vb.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("first Close = %v, want ErrInvalidValue", err)
	}
	if rec.released != 1 || rec.freed != 0 {
		t.Errorf("after failed close: released=%d freed=%d, want 1, 0", rec.released, rec.freed)
	}

	unmapFails.Store(false)
	if err := vb.Close(); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
	if rec.unmapped != 1 || rec.freed != 1 || rec.released != 1 {
		t.Errorf("after retry: unmap=%d free=%d release=%d, want 1,1,1", rec.unmapped, rec.freed, rec.released)
	}
}

func TestAllocVirtualPartialSymbols(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	d.CuMemUnmap = nil
	ctx := newManagedContext(t, d)
	if _, err := AllocVirtual[float32](ctx, 8); !errors.Is(err, ErrSymbolUnavailable) {
		t.Fatalf("AllocVirtual = %v, want ErrSymbolUnavailable", err)
	}
	if rec.created != 0 || rec.reserved != 0 {
		t.Errorf("nothing should be acquired with a missing symbol: created=%d reserved=%d", rec.created, rec.reserved)
	}
}

func TestRoundUp(t *testing.T) {
	cases := []struct {
		v, mult, want uint64
		ok            bool
	}{
		{0, 8, 0, true},
		{1, 8, 8, true},
		{8, 8, 8, true},
		{9, 8, 16, true},
		{4096, 2 << 20, 2 << 20, true},
		{math.MaxUint64, 8, 0, false},
		{math.MaxUint64 - 3, 8, 0, false},
	}
	for _, c := range cases {
		got, ok := roundUp(c.v, c.mult)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("roundUp(%d, %d) = %d, %v; want %d, %v", c.v, c.mult, got, ok, c.want, c.ok)
		}
	}
}

func TestAllocVirtualRollbackUnmapAlsoFails(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	d.CuMemSetAccess = func(cudasys.CUdeviceptr, uint64, *cudasys.CUmemAccessDesc, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	d.CuMemUnmap = func(cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	ctx := newManagedContext(t, d)
	if _, err := AllocVirtual[float32](ctx, 1024); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("AllocVirtual = %v, want ErrInvalidValue", err)
	}
	if rec.freed != 0 || rec.released != 1 {
		t.Errorf("with failed unmap the address must not be freed: freed=%d released=%d, want 0, 1", rec.freed, rec.released)
	}
}

func TestVirtualBufferClosePartialReleaseFail(t *testing.T) {
	var rec vmmRec
	d := vmmDriver(&rec)
	var releaseFails atomic.Bool
	releaseFails.Store(true)
	d.CuMemRelease = func(cudasys.CUmemGenericAllocationHandle) cudasys.CUresult {
		if releaseFails.Load() {
			return cudasys.CUDA_ERROR_INVALID_VALUE
		}
		rec.released++
		return cudasys.CUDA_SUCCESS
	}
	ctx := newManagedContext(t, d)
	vb, err := AllocVirtual[float32](ctx, 1024)
	if err != nil {
		t.Fatalf("AllocVirtual: %v", err)
	}

	if err := vb.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("first Close = %v, want ErrInvalidValue", err)
	}
	if rec.unmapped != 1 || rec.freed != 1 {
		t.Errorf("unmap and address-free should still run: unmap=%d freed=%d", rec.unmapped, rec.freed)
	}
	if err := vb.CopyFrom(context.Background(), make([]float32, 1024)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyFrom after partial teardown = %v, want ErrBufferClosed", err)
	}

	releaseFails.Store(false)
	if err := vb.Close(); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
	if rec.released != 1 {
		t.Errorf("released = %d, want 1", rec.released)
	}
}
