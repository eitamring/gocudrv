package cuda

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ebitengine/purego"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/hostcb"
)

func hostFuncFixture(t *testing.T, launch func(cudasys.CUstream, uintptr, uintptr) cudasys.CUresult) (*Context, *Stream) {
	t.Helper()
	drv := fakeMemoryDriver(&memCalls{}, 0x90000)
	drv.CuStreamCreate = func(s *cudasys.CUstream, _ uint32) cudasys.CUresult { *s = 0x5151; return cudasys.CUDA_SUCCESS }
	drv.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuLaunchHostFunc = launch
	ctx := newTestContext(t, drv)
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	return ctx, stream
}

func TestLaunchHostFuncRunsThroughDriverCallback(t *testing.T) {
	var ran atomic.Bool
	_, stream := hostFuncFixture(t, func(st cudasys.CUstream, fn uintptr, userData uintptr) cudasys.CUresult {
		if st != 0x5151 {
			t.Errorf("stream = %#x, want 0x5151", st)
		}
		purego.SyscallN(fn, userData)
		return cudasys.CUDA_SUCCESS
	})
	if err := stream.LaunchHostFunc(func() { ran.Store(true) }); err != nil {
		t.Fatalf("LaunchHostFunc: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for !ran.Load() && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if !ran.Load() {
		t.Fatal("host function never ran through the driver-style callback")
	}
}

func TestLaunchHostFuncRejects(t *testing.T) {
	_, stream := hostFuncFixture(t, func(cudasys.CUstream, uintptr, uintptr) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	})
	var nilStream *Stream
	if err := nilStream.LaunchHostFunc(func() {}); !errors.Is(err, ErrNilStream) {
		t.Errorf("nil stream = %v, want ErrNilStream", err)
	}
	if err := stream.LaunchHostFunc(nil); !errors.Is(err, ErrNilHostFunc) {
		t.Errorf("nil fn = %v, want ErrNilHostFunc", err)
	}
	closed, err := stream.ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	_ = closed.Close()
	if err := closed.LaunchHostFunc(func() {}); !errors.Is(err, ErrStreamClosed) {
		t.Errorf("closed stream = %v, want ErrStreamClosed", err)
	}
}

func TestLaunchHostFuncSymbolUnavailable(t *testing.T) {
	ctx, stream := hostFuncFixture(t, nil)
	ctx.driver.CuLaunchHostFunc = nil
	if err := stream.LaunchHostFunc(func() {}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing symbol = %v, want ErrSymbolUnavailable", err)
	}
}

func TestLaunchHostFuncUnregistersOnFailure(t *testing.T) {
	var gotKey atomic.Uintptr
	_, stream := hostFuncFixture(t, func(_ cudasys.CUstream, _ uintptr, userData uintptr) cudasys.CUresult {
		gotKey.Store(userData)
		return cudasys.CUDA_ERROR_INVALID_VALUE
	})
	if err := stream.LaunchHostFunc(func() {}); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("err = %v, want ErrInvalidValue", err)
	}
	if hostcb.Pending(gotKey.Load()) {
		t.Fatal("failed enqueue left the callback registered")
	}
}

func TestLaunchHostFuncClosedContext(t *testing.T) {
	ctx, stream := hostFuncFixture(t, func(cudasys.CUstream, uintptr, uintptr) cudasys.CUresult {
		t.Error("driver must not be reached after context close")
		return cudasys.CUDA_SUCCESS
	})
	if err := ctx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := stream.LaunchHostFunc(func() { t.Error("must not run") }); !errors.Is(err, ErrContextClosed) {
		t.Errorf("closed context = %v, want ErrContextClosed", err)
	}
}
