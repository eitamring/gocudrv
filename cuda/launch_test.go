package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

type launchFake struct {
	launchCalls atomic.Int32
	params      []unsafe.Pointer
}

func (l *launchFake) driver(t testing.TB) *cudasys.Driver {
	t.Helper()
	return &cudasys.Driver{
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
		CuMemAlloc: func(p *cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			*p = 0xDEAD
			return cudasys.CUDA_SUCCESS
		},
		CuMemFree: func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleLoadData: func(mod *cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*mod = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
		CuModuleUnload: func(cudasys.CUmodule) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleGetFunction: func(fn *cudasys.CUfunction, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*fn = 0xCAFE
			return cudasys.CUDA_SUCCESS
		},
		CuLaunchKernel: func(
			fn cudasys.CUfunction,
			gridX, gridY, gridZ uint32,
			blockX, blockY, blockZ uint32,
			sharedMemBytes uint32,
			stream cudasys.CUstream,
			params *unsafe.Pointer,
			extra *unsafe.Pointer,
		) cudasys.CUresult {
			l.launchCalls.Add(1)
			if fn != 0xCAFE {
				t.Errorf("fn = %#x, want 0xCAFE", fn)
			}
			if gridX != 4 || gridY != 1 || gridZ != 1 {
				t.Errorf("grid = (%d,%d,%d), want (4,1,1)", gridX, gridY, gridZ)
			}
			if blockX != 256 || blockY != 1 || blockZ != 1 {
				t.Errorf("block = (%d,%d,%d), want (256,1,1)", blockX, blockY, blockZ)
			}
			if sharedMemBytes != 32 {
				t.Errorf("shared = %d, want 32", sharedMemBytes)
			}
			if stream != 0 {
				t.Errorf("stream = %#x, want default stream", stream)
			}
			if extra != nil {
				t.Errorf("extra = %p, want nil", extra)
			}
			l.params = append([]unsafe.Pointer(nil), unsafe.Slice(params, 5)...)
			return cudasys.CUDA_SUCCESS
		},
	}
}

func BenchmarkFunctionLaunch(b *testing.B) {
	var l launchFake
	resetDriver()
	mu.Lock()
	driver = l.driver(b)
	mu.Unlock()
	b.Cleanup(resetDriver)
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	b.Cleanup(func() { _ = ctx.Close() })
	mod, _ := ctx.LoadModule([]byte{'P', 0})
	b.Cleanup(func() { _ = mod.Close() })
	fn, _ := mod.Function("k")
	buf, _ := Alloc[float32](ctx, 4)
	b.Cleanup(func() { _ = buf.Close() })
	cfg := LaunchConfig1D(1024, 256)
	cfg.SharedMemBytes = 32

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := fn.Launch(context.Background(), cfg,
			Arg(buf),
			ArgValue(int32(i)),
			ArgValue(uint32(i)),
			ArgValue(float32(i)),
		); err != nil {
			b.Fatal(err)
		}
	}
}

func newLaunchFixture(t *testing.T) (*Context, *Module, *Function, *Buffer[float32]) {
	t.Helper()
	var l launchFake
	installDriver(t, l.driver(t))
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("k")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	return ctx, mod, fn, buf
}

func TestLaunchConfig1D(t *testing.T) {
	cases := []struct {
		name string
		n    int
		bs   int
		want LaunchConfig
	}{
		{"exact", 1024, 256, LaunchConfig{GridX: 4, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}},
		{"round up", 1025, 256, LaunchConfig{GridX: 5, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}},
		{"zero n", 0, 256, LaunchConfig{}},
		{"zero block", 1024, 0, LaunchConfig{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := LaunchConfig1D(tc.n, tc.bs); got != tc.want {
				t.Errorf("LaunchConfig1D = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestFunctionLaunchPacksArgs(t *testing.T) {
	var l launchFake
	installDriver(t, l.driver(t))
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })
	mod, _ := ctx.LoadModule([]byte{'P', 0})
	t.Cleanup(func() { _ = mod.Close() })
	fn, _ := mod.Function("k")
	buf, _ := Alloc[float32](ctx, 4)
	t.Cleanup(func() { _ = buf.Close() })

	cfg := LaunchConfig1D(1024, 256)
	cfg.SharedMemBytes = 32
	err := fn.Launch(context.Background(), cfg,
		Arg(buf),
		ArgValue(int32(-3)),
		ArgValue(uint32(7)),
		ArgValue(float32(1.5)),
		ArgValue(float64(2.5)),
	)
	if err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if l.launchCalls.Load() != 1 {
		t.Fatalf("launch calls = %d, want 1", l.launchCalls.Load())
	}
	if got := *(*cudasys.CUdeviceptr)(l.params[0]); got != 0xDEAD {
		t.Errorf("arg0 = %#x, want 0xDEAD", got)
	}
	if got := *(*int32)(l.params[1]); got != -3 {
		t.Errorf("arg1 = %d, want -3", got)
	}
	if got := *(*uint32)(l.params[2]); got != 7 {
		t.Errorf("arg2 = %d, want 7", got)
	}
	if got := *(*float32)(l.params[3]); got != 1.5 {
		t.Errorf("arg3 = %v, want 1.5", got)
	}
	if got := *(*float64)(l.params[4]); got != 2.5 {
		t.Errorf("arg4 = %v, want 2.5", got)
	}
}

func TestFunctionLaunchRejects(t *testing.T) {
	ctx, mod, fn, buf := newLaunchFixture(t)
	_ = ctx

	otherDriver := (&launchFake{}).driver(t)
	installDriver(t, otherDriver)
	otherDev, _ := GetDevice(0)
	otherCtx, _ := otherDev.Primary()
	t.Cleanup(func() { _ = otherCtx.Close() })
	otherBuf, _ := Alloc[float32](otherCtx, 4)
	t.Cleanup(func() { _ = otherBuf.Close() })

	closedBuf, _ := Alloc[float32](mod.ctx, 4)
	if err := closedBuf.Close(); err != nil {
		t.Fatalf("close buffer: %v", err)
	}

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil function",
			func() error {
				var f *Function
				return f.Launch(context.Background(), LaunchConfig1D(1, 1))
			},
			ErrNilFunction,
		},
		{"invalid config", func() error { return fn.Launch(context.Background(), LaunchConfig{}) }, ErrInvalidLaunchConfig},
		{"nil arg", func() error { return fn.Launch(context.Background(), LaunchConfig1D(1, 1), nil) }, ErrNilKernelArg},
		{"nil buffer", func() error { return fn.Launch(context.Background(), LaunchConfig1D(1, 1), Arg[float32](nil)) }, ErrNilBuffer},
		{"closed buffer", func() error { return fn.Launch(context.Background(), LaunchConfig1D(1, 1), Arg(closedBuf)) }, ErrBufferClosed},
		{"wrong context", func() error { return fn.Launch(context.Background(), LaunchConfig1D(1, 1), Arg(otherBuf)) }, ErrContextMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}

	if err := mod.Close(); err != nil {
		t.Fatalf("close module: %v", err)
	}
	if err := fn.Launch(context.Background(), LaunchConfig1D(1, 1), Arg(buf)); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("closed module err = %v, want ErrModuleClosed", err)
	}
}

func TestFunctionLaunchCanceledBeforeSubmit(t *testing.T) {
	_, _, fn, _ := newLaunchFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := fn.Launch(ctx, LaunchConfig1D(1, 1)); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestFunctionLaunchHoldsModuleAndBufferLocksDuringCall(t *testing.T) {
	launchEntered := make(chan struct{})
	mayFinish := make(chan struct{})
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
		CuMemAlloc: func(p *cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			*p = 0xDEAD
			return cudasys.CUDA_SUCCESS
		},
		CuMemFree: func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleLoadData: func(mod *cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*mod = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
		CuModuleUnload: func(cudasys.CUmodule) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleGetFunction: func(fn *cudasys.CUfunction, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*fn = 0xCAFE
			return cudasys.CUDA_SUCCESS
		},
		CuLaunchKernel: func(
			cudasys.CUfunction,
			uint32, uint32, uint32,
			uint32, uint32, uint32,
			uint32,
			cudasys.CUstream,
			*unsafe.Pointer,
			*unsafe.Pointer,
		) cudasys.CUresult {
			close(launchEntered)
			<-mayFinish
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })
	mod, _ := ctx.LoadModule([]byte{'P', 0})
	fn, _ := mod.Function("k")
	buf, _ := Alloc[float32](ctx, 4)

	launchDone := make(chan error, 1)
	go func() {
		launchDone <- fn.Launch(context.Background(), LaunchConfig1D(1, 1), Arg(buf))
	}()
	select {
	case <-launchEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("launch did not enter driver")
	}

	closeModuleDone := make(chan error, 1)
	closeBufferDone := make(chan error, 1)
	go func() { closeModuleDone <- mod.Close() }()
	go func() { closeBufferDone <- buf.Close() }()

	select {
	case err := <-closeModuleDone:
		t.Fatalf("module close returned during launch: %v", err)
	case err := <-closeBufferDone:
		t.Fatalf("buffer close returned during launch: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	close(mayFinish)
	if err := <-launchDone; err != nil {
		t.Errorf("Launch: %v", err)
	}
	if err := <-closeModuleDone; err != nil {
		t.Errorf("module close: %v", err)
	}
	if err := <-closeBufferDone; err != nil {
		t.Errorf("buffer close: %v", err)
	}
}
