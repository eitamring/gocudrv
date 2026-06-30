package cuda

import (
	"context"
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

// managedDriver is a fake driver with the context, stream, and free machinery
// the managed-memory tests need. Tests set the managed entry points themselves.
func managedDriver() *cudasys.Driver {
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet:      func(d *cudasys.CUdevice, _ int32) cudasys.CUresult { *d = 0; return cudasys.CUDA_SUCCESS },
		CuDevicePrimaryCtxRetain: func(c *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*c = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSetCurrent:           func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemFree:                 func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamCreate:            func(s *cudasys.CUstream, _ uint32) cudasys.CUresult { *s = 0x5151; return cudasys.CUDA_SUCCESS },
		CuStreamDestroy:           func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
	}
}

func newManagedContext(t *testing.T, d *cudasys.Driver) *Context {
	t.Helper()
	installDriver(t, d)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	return ctx
}

func TestAllocManaged(t *testing.T) {
	d := managedDriver()
	storage := make([]byte, 16*4)
	var got struct {
		n     uint64
		flags uint32
	}
	d.CuMemAllocManaged = func(pp **byte, n uint64, flags uint32) cudasys.CUresult {
		got.n, got.flags = n, flags
		*pp = &storage[0]
		return cudasys.CUDA_SUCCESS
	}
	ctx := newManagedContext(t, d)

	mb, err := AllocManaged[float32](ctx, 16)
	if err != nil {
		t.Fatalf("AllocManaged: %v", err)
	}
	if got.n != 64 || got.flags != memAttachGlobal {
		t.Errorf("driver got n=%d flags=%d, want 64, %d", got.n, got.flags, memAttachGlobal)
	}
	if mb.Len() != 16 || mb.Bytes() != 64 || mb.DevicePtr() == 0 {
		t.Errorf("len=%d bytes=%d ptr=%#x", mb.Len(), mb.Bytes(), mb.DevicePtr())
	}

	// The host slice aliases the managed allocation: a write is visible on reread.
	s := mb.Slice()
	if len(s) != 16 {
		t.Fatalf("Slice len = %d, want 16", len(s))
	}
	s[3] = 2.5
	if mb.Slice()[3] != 2.5 {
		t.Errorf("Slice does not alias the managed memory")
	}

	if err := mb.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := mb.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if mb.Slice() != nil {
		t.Errorf("Slice after Close should be nil")
	}
	if err := mb.Advise(AdviseSetReadMostly); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("Advise after close = %v, want ErrBufferClosed", err)
	}
}

func TestAllocManagedRejects(t *testing.T) {
	if _, err := AllocManaged[float32](nil, 8); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	ctx := newManagedContext(t, managedDriver()) // CuMemAllocManaged left nil
	if _, err := AllocManaged[float32](ctx, 0); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero n = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocManaged[float32](ctx, 8); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("unavailable = %v, want ErrSymbolUnavailable", err)
	}
}

func TestManagedPrefetch(t *testing.T) {
	d := managedDriver()
	storage := make([]byte, 64)
	d.CuMemAllocManaged = func(pp **byte, _ uint64, _ uint32) cudasys.CUresult { *pp = &storage[0]; return cudasys.CUDA_SUCCESS }
	var got struct {
		ptr    cudasys.CUdeviceptr
		count  uint64
		dst    cudasys.CUdevice
		stream cudasys.CUstream
	}
	d.CuMemPrefetchAsync = func(ptr cudasys.CUdeviceptr, count uint64, dst cudasys.CUdevice, s cudasys.CUstream) cudasys.CUresult {
		got.ptr, got.count, got.dst, got.stream = ptr, count, dst, s
		return cudasys.CUDA_SUCCESS
	}
	ctx := newManagedContext(t, d)
	mb, _ := AllocManaged[float32](ctx, 16)
	t.Cleanup(func() { _ = mb.Close() })
	stream, _ := ctx.NewStream()
	t.Cleanup(func() { _ = stream.Close() })

	if err := mb.PrefetchToDevice(context.Background(), stream); err != nil {
		t.Fatalf("PrefetchToDevice: %v", err)
	}
	if got.dst != 0 || got.count != 64 || got.ptr != mb.DevicePtr() || got.stream != 0x5151 {
		t.Errorf("to-device got %+v, want dst=0 count=64 ptr=%#x stream=0x5151", got, mb.DevicePtr())
	}
	if err := mb.PrefetchToHost(context.Background(), stream); err != nil {
		t.Fatalf("PrefetchToHost: %v", err)
	}
	if got.dst != deviceCPU {
		t.Errorf("to-host dst = %d, want %d (CU_DEVICE_CPU)", got.dst, deviceCPU)
	}
	if err := mb.PrefetchToDevice(context.Background(), nil); !errors.Is(err, ErrNilStream) {
		t.Errorf("nil stream = %v, want ErrNilStream", err)
	}
	var nilMB *ManagedBuffer[float32]
	if err := nilMB.PrefetchToDevice(context.Background(), nil); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil receiver to-device = %v, want ErrNilBuffer", err)
	}
	if err := nilMB.PrefetchToHost(context.Background(), nil); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil receiver to-host = %v, want ErrNilBuffer", err)
	}
}

func TestManagedAdvise(t *testing.T) {
	d := managedDriver()
	storage := make([]byte, 64)
	d.CuMemAllocManaged = func(pp **byte, _ uint64, _ uint32) cudasys.CUresult { *pp = &storage[0]; return cudasys.CUDA_SUCCESS }
	var got struct {
		ptr    cudasys.CUdeviceptr
		count  uint64
		advice int32
		dev    cudasys.CUdevice
	}
	d.CuMemAdvise = func(ptr cudasys.CUdeviceptr, count uint64, advice int32, dev cudasys.CUdevice) cudasys.CUresult {
		got.ptr, got.count, got.advice, got.dev = ptr, count, advice, dev
		return cudasys.CUDA_SUCCESS
	}
	ctx := newManagedContext(t, d)
	mb, _ := AllocManaged[float32](ctx, 16)
	t.Cleanup(func() { _ = mb.Close() })

	if err := mb.Advise(AdviseSetPreferredLocation); err != nil {
		t.Fatalf("Advise: %v", err)
	}
	if got.advice != int32(AdviseSetPreferredLocation) || got.count != 64 || got.ptr != mb.DevicePtr() || got.dev != 0 {
		t.Errorf("got %+v, want advice=3 count=64 ptr=%#x dev=0", got, mb.DevicePtr())
	}

	var nilMB *ManagedBuffer[float32]
	if err := nilMB.Advise(AdviseSetReadMostly); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil receiver = %v, want ErrNilBuffer", err)
	}
}
