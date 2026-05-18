package cuda

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/eitamring/gocudrv/cudasys"
)

type ctxCalls struct {
	retain     atomic.Int32
	release    atomic.Int32
	setCurrent atomic.Int32
	sync       atomic.Int32
	priority   atomic.Int32
	lastSetCtx atomic.Uintptr
}

func fakeContextDriver(c *ctxCalls, ctxHandle cudasys.CUcontext) *cudasys.Driver {
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			c.retain.Add(1)
			*ctx = ctxHandle
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			c.release.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(ctx cudasys.CUcontext) cudasys.CUresult {
			c.setCurrent.Add(1)
			c.lastSetCtx.Store(uintptr(ctx))
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSynchronize: func() cudasys.CUresult {
			c.sync.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuCtxGetStreamPriorityRange: func(least, greatest *int32) cudasys.CUresult {
			c.priority.Add(1)
			*least = 0
			*greatest = -2
			return cudasys.CUDA_SUCCESS
		},
	}
}

func TestPrimarySuccess(t *testing.T) {
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("get device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	if calls.retain.Load() != 1 {
		t.Errorf("retain calls = %d, want 1", calls.retain.Load())
	}
	if calls.setCurrent.Load() != 1 {
		t.Errorf("set-current calls = %d, want 1", calls.setCurrent.Load())
	}
	if got := calls.lastSetCtx.Load(); got != 0xC0FFEE {
		t.Errorf("set-current arg = %#x, want 0xC0FFEE", got)
	}
	if ctx.Device() != dev {
		t.Error("Device back-reference does not match")
	}
}

func TestPrimaryRetainFailure(t *testing.T) {
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(*cudasys.CUcontext, cudasys.CUdevice) cudasys.CUresult {
			return cudasys.CUDA_ERROR_OUT_OF_MEMORY
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			t.Error("Release must not run when Retain failed")
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(cudasys.CUcontext) cudasys.CUresult {
			t.Error("SetCurrent must not run when Retain failed")
			return cudasys.CUDA_SUCCESS
		},
	})

	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("get device: %v", err)
	}
	if _, err := dev.Primary(); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestPrimarySetCurrentFailureReleases(t *testing.T) {
	releaseCalls := atomic.Int32{}
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xDEAD
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(cudasys.CUcontext) cudasys.CUresult {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			releaseCalls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	})

	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("get device: %v", err)
	}
	if _, err := dev.Primary(); !errors.Is(err, ErrInvalidContext) {
		t.Errorf("err = %v, want ErrInvalidContext", err)
	}
	if releaseCalls.Load() != 1 {
		t.Errorf("release calls = %d, want 1 (rollback)", releaseCalls.Load())
	}
}

func TestPrimarySetCurrentFailureReportsReleaseFailure(t *testing.T) {
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xDEAD
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(cudasys.CUcontext) cudasys.CUresult {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			return cudasys.CUDA_ERROR_INVALID_DEVICE
		},
	})

	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("get device: %v", err)
	}
	_, err = dev.Primary()
	if !errors.Is(err, ErrInvalidContext) {
		t.Errorf("err = %v, want ErrInvalidContext", err)
	}
	if !errors.Is(err, ErrInvalidDevice) {
		t.Errorf("err = %v, want ErrInvalidDevice from rollback release", err)
	}
}

func TestPrimaryNilDevice(t *testing.T) {
	installDriver(t, &cudasys.Driver{})
	var dev *Device
	if _, err := dev.Primary(); !errors.Is(err, ErrNilDevice) {
		t.Errorf("err = %v, want ErrNilDevice", err)
	}
}

func TestSynchronize(t *testing.T) {
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if calls.sync.Load() != 1 {
		t.Errorf("sync calls = %d, want 1", calls.sync.Load())
	}
}

func TestStreamPriorityRange(t *testing.T) {
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	least, greatest, err := ctx.StreamPriorityRange()
	if err != nil {
		t.Fatalf("StreamPriorityRange: %v", err)
	}
	if least != 0 || greatest != -2 {
		t.Errorf("range = (%d, %d), want (0, -2)", least, greatest)
	}
	if calls.priority.Load() != 1 {
		t.Errorf("priority calls = %d, want 1", calls.priority.Load())
	}

	var nilCtx *Context
	if _, _, err := nilCtx.StreamPriorityRange(); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil context err = %v, want ErrNilContext", err)
	}
}

func TestCloseReleasesAndClearsCurrent(t *testing.T) {
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}

	if err := ctx.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	// One SetCurrent for Primary (set to ctx), one for Close (set to 0).
	if calls.setCurrent.Load() != 2 {
		t.Errorf("set-current calls = %d, want 2", calls.setCurrent.Load())
	}
	if got := calls.lastSetCtx.Load(); got != 0 {
		t.Errorf("final set-current arg = %#x, want 0", got)
	}
	if calls.release.Load() != 1 {
		t.Errorf("release calls = %d, want 1", calls.release.Load())
	}
}

func TestCloseReleasesWhenClearCurrentFails(t *testing.T) {
	releaseCalls := atomic.Int32{}
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(ctx cudasys.CUcontext) cudasys.CUresult {
			if ctx == 0 {
				return cudasys.CUDA_ERROR_INVALID_CONTEXT
			}
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			releaseCalls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	})

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	if err := ctx.Close(); !errors.Is(err, ErrInvalidContext) {
		t.Errorf("close err = %v, want ErrInvalidContext", err)
	}
	if releaseCalls.Load() != 1 {
		t.Errorf("release calls = %d, want 1", releaseCalls.Load())
	}
}

func TestCloseWaitsForInFlightContextWorkBeforeRelease(t *testing.T) {
	releaseCalls := atomic.Int32{}
	syncStarted := make(chan struct{})
	syncFinish := make(chan struct{})
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			releaseCalls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSynchronize: func() cudasys.CUresult {
			close(syncStarted)
			<-syncFinish
			return cudasys.CUDA_SUCCESS
		},
	})

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}

	syncDone := make(chan error, 1)
	go func() { syncDone <- ctx.Synchronize(context.Background()) }()
	<-syncStarted

	closeDone := make(chan error, 1)
	go func() { closeDone <- ctx.Close() }()

	select {
	case err := <-closeDone:
		t.Fatalf("Close returned before in-flight sync finished: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	if releaseCalls.Load() != 0 {
		t.Fatalf("release ran while sync was still in flight")
	}

	close(syncFinish)
	if err := <-syncDone; err != nil {
		t.Fatalf("sync: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Fatalf("close: %v", err)
	}
	if releaseCalls.Load() != 1 {
		t.Errorf("release calls = %d, want 1", releaseCalls.Load())
	}
}

func TestCloseReportsReleaseFailure(t *testing.T) {
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			return cudasys.CUDA_ERROR_INVALID_DEVICE
		},
	})

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	if err := ctx.Close(); !errors.Is(err, ErrInvalidDevice) {
		t.Errorf("close err = %v, want ErrInvalidDevice", err)
	}
}

func TestCloseIdempotent(t *testing.T) {
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}

	if err := ctx.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Errorf("second close: %v", err)
	}
	if calls.release.Load() != 1 {
		t.Errorf("release calls = %d, want 1 (second Close should be a no-op)", calls.release.Load())
	}
}

func TestNilContextMethods(t *testing.T) {
	var ctx *Context
	if got := ctx.Device(); got != nil {
		t.Errorf("Device = %v, want nil", got)
	}
	if err := ctx.Synchronize(context.Background()); !errors.Is(err, ErrNilContext) {
		t.Errorf("Synchronize err = %v, want ErrNilContext", err)
	}
	if err := ctx.Close(); !errors.Is(err, ErrNilContext) {
		t.Errorf("Close err = %v, want ErrNilContext", err)
	}

	ctx = &Context{}
	if err := ctx.Synchronize(context.Background()); !errors.Is(err, ErrNilContext) {
		t.Errorf("zero Context Synchronize err = %v, want ErrNilContext", err)
	}
	if err := ctx.Close(); !errors.Is(err, ErrNilContext) {
		t.Errorf("zero Context Close err = %v, want ErrNilContext", err)
	}
}

func TestMethodsAfterClose(t *testing.T) {
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	if err := ctx.Synchronize(context.Background()); !errors.Is(err, ErrContextClosed) {
		t.Errorf("Synchronize after close: err = %v, want ErrContextClosed", err)
	}
}

func TestSynchronizeContextCanceled(t *testing.T) {
	// Synchronize that takes a long time on the executor side; ctx cancels
	// before the result arrives. Synchronize should return ctx.Err() while
	// the GPU work (here: a sleep on the executor) continues underneath.
	driverCalled := atomic.Int32{}
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSetCurrent:           func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSynchronize: func() cudasys.CUresult {
			driverCalled.Add(1)
			time.Sleep(50 * time.Millisecond)
			return cudasys.CUDA_SUCCESS
		},
	})

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	cctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()

	syncErr := ctx.Synchronize(cctx)
	if !errors.Is(syncErr, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", syncErr)
	}
	// Wait for the abandoned call to finish so cleanup is clean.
	deadline := time.Now().Add(time.Second)
	for driverCalled.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if driverCalled.Load() != 1 {
		t.Errorf("CuCtxSynchronize calls = %d, want 1 (still runs after ctx cancel)", driverCalled.Load())
	}
}

func TestSynchronizeBeforeClose(t *testing.T) {
	// Sequential, not concurrent: many syncs all succeed, then Close once.
	var calls ctxCalls
	installDriver(t, fakeContextDriver(&calls, 0xC0FFEE))

	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}

	var wg sync.WaitGroup
	const n = 20
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if err := ctx.Synchronize(context.Background()); err != nil {
				t.Errorf("sync: %v", err)
			}
		}()
	}
	wg.Wait()
	if calls.sync.Load() != n {
		t.Errorf("sync calls = %d, want %d", calls.sync.Load(), n)
	}
	if err := ctx.Close(); err != nil {
		t.Errorf("close: %v", err)
	}
}
