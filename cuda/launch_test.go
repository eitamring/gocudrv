package cuda

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

type launchFake struct {
	launchCalls    atomic.Int32
	params         []unsafe.Pointer
	expectedStream cudasys.CUstream
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
			if stream != l.expectedStream {
				t.Errorf("stream = %#x, want %#x", stream, l.expectedStream)
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

func TestFunctionLaunchOnUsesStream(t *testing.T) {
	var l launchFake
	l.expectedStream = 0x5151
	drv := l.driver(t)
	drv.CuStreamCreate = func(stream *cudasys.CUstream, flags uint32) cudasys.CUresult {
		if flags != streamNonBlocking {
			t.Errorf("flags = %d, want %d", flags, streamNonBlocking)
		}
		*stream = 0x5151
		return cudasys.CUDA_SUCCESS
	}
	drv.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	installDriver(t, drv)

	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })
	stream, _ := ctx.NewStream()
	t.Cleanup(func() { _ = stream.Close() })
	mod, _ := ctx.LoadModule([]byte{'P', 0})
	t.Cleanup(func() { _ = mod.Close() })
	fn, _ := mod.Function("k")
	buf, _ := Alloc[float32](ctx, 4)
	t.Cleanup(func() { _ = buf.Close() })

	cfg := LaunchConfig1D(1024, 256)
	cfg.SharedMemBytes = 32
	if err := fn.LaunchOn(context.Background(), stream, cfg, Arg(buf)); err != nil {
		t.Fatalf("LaunchOn: %v", err)
	}
	if l.launchCalls.Load() != 1 {
		t.Errorf("launch calls = %d, want 1", l.launchCalls.Load())
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
	streamDriver := (&launchFake{}).driver(t)
	streamDriver.CuStreamCreate = func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
		*stream = 0x5151
		return cudasys.CUDA_SUCCESS
	}
	streamDriver.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	installDriver(t, streamDriver)
	streamCtxDev, _ := GetDevice(0)
	streamCtx, _ := streamCtxDev.Primary()
	t.Cleanup(func() { _ = streamCtx.Close() })
	stream, _ := streamCtx.NewStream()
	t.Cleanup(func() { _ = stream.Close() })
	closedStream, _ := streamCtx.NewStream()
	if err := closedStream.Close(); err != nil {
		t.Fatalf("close stream: %v", err)
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
		{"nil stream", func() error { return fn.LaunchOn(context.Background(), nil, LaunchConfig1D(1, 1)) }, ErrNilStream},
		{"closed stream", func() error { return fn.LaunchOn(context.Background(), closedStream, LaunchConfig1D(1, 1)) }, ErrStreamClosed},
		{"wrong stream context", func() error { return fn.LaunchOn(context.Background(), stream, LaunchConfig1D(1, 1)) }, ErrContextMismatch},
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

func TestFunctionLaunchOnHoldsStreamLockDuringCall(t *testing.T) {
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
		CuStreamCreate: func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
			*stream = 0x5151
			return cudasys.CUDA_SUCCESS
		},
		CuStreamDestroy: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
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
	stream, _ := ctx.NewStream()
	mod, _ := ctx.LoadModule([]byte{'P', 0})
	fn, _ := mod.Function("k")

	launchDone := make(chan error, 1)
	go func() {
		launchDone <- fn.LaunchOn(context.Background(), stream, LaunchConfig1D(1, 1))
	}()
	select {
	case <-launchEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("launch did not enter driver")
	}

	closeStreamDone := make(chan error, 1)
	go func() { closeStreamDone <- stream.Close() }()
	select {
	case err := <-closeStreamDone:
		t.Fatalf("stream close returned during launch: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	close(mayFinish)
	if err := <-launchDone; err != nil {
		t.Errorf("LaunchOn: %v", err)
	}
	if err := <-closeStreamDone; err != nil {
		t.Errorf("stream close: %v", err)
	}
	_ = mod.Close()
}

func newEscapeFixture(t *testing.T) (*Module, *Function, *atomic.Int32, *atomic.Uint64) {
	t.Helper()
	var launches atomic.Int32
	var firstParam atomic.Uint64
	drv := (&launchFake{}).driver(t)
	drv.CuLaunchKernel = func(_ cudasys.CUfunction, _, _, _, _, _, _, _ uint32, _ cudasys.CUstream, params *unsafe.Pointer, _ *unsafe.Pointer) cudasys.CUresult {
		launches.Add(1)
		if params != nil {
			firstParam.Store(*(*uint64)(*params))
		}
		return cudasys.CUDA_SUCCESS
	}
	installDriver(t, drv)
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
	return mod, fn, &launches, &firstParam
}

func TestLaunchArgDevicePtr(t *testing.T) {
	_, fn, launches, firstParam := newEscapeFixture(t)
	if err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), ArgDevicePtr(0xABCD)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if launches.Load() != 1 {
		t.Errorf("launches = %d, want 1", launches.Load())
	}
	if firstParam.Load() != 0xABCD {
		t.Errorf("packed device pointer = %#x, want 0xabcd", firstParam.Load())
	}
}

func TestLaunchArgRaw(t *testing.T) {
	_, fn, launches, firstParam := newEscapeFixture(t)
	v := uint32(0xCAFE)
	if err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), ArgRaw(unsafe.Pointer(&v), 4)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if launches.Load() != 1 {
		t.Errorf("launches = %d, want 1", launches.Load())
	}
	if firstParam.Load() != 0xCAFE {
		t.Errorf("packed raw value = %#x, want 0xcafe", firstParam.Load())
	}
}

func TestLaunchArgRawRejects(t *testing.T) {
	_, fn, launches, _ := newEscapeFixture(t)
	v := uint32(1)
	cases := []struct {
		name string
		arg  KernelArg
		want error
	}{
		{"nil value", ArgRaw(nil, 4), ErrNilKernelArg},
		{"zero size", ArgRaw(unsafe.Pointer(&v), 0), ErrInvalidArgSize},
		{"too big", ArgRaw(unsafe.Pointer(&v), maxRawArgBytes+1), ErrInvalidArgSize},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), tc.arg); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
	if launches.Load() != 0 {
		t.Errorf("a rejected launch reached the driver: %d", launches.Load())
	}
}

func TestLaunchManyArgsSpill(t *testing.T) {
	_, fn, launches, _ := newEscapeFixture(t)
	args := make([]KernelArg, 20)
	for i := range args {
		args[i] = ArgValue(int32(i))
	}
	if err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), args...); err != nil {
		t.Fatalf("Launch with 20 args: %v", err)
	}
	var big [16]byte
	big[0] = 0x42
	if err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), ArgRaw(unsafe.Pointer(&big), 16)); err != nil {
		t.Fatalf("Launch with 16-byte raw arg: %v", err)
	}
	if launches.Load() != 2 {
		t.Errorf("launches = %d, want 2", launches.Load())
	}
}

func TestLaunchModuleClosed(t *testing.T) {
	mod, fn, launches, _ := newEscapeFixture(t)
	if err := mod.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), ArgDevicePtr(0xABCD)); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("launch after close = %v, want ErrModuleClosed", err)
	}
	if launches.Load() != 0 {
		t.Error("launch reached the driver after module close")
	}
}

func TestLaunchModuleCloseRace(t *testing.T) {
	mod, fn, _, _ := newEscapeFixture(t)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := fn.Launch(context.Background(), LaunchConfig1D(256, 256), ArgDevicePtr(0xABCD))
			if err != nil && !errors.Is(err, ErrModuleClosed) {
				t.Errorf("unexpected launch error: %v", err)
			}
		}()
	}
	_ = mod.Close()
	wg.Wait()
}

func TestLaunchPacked(t *testing.T) {
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

	p, err := Pack(
		Arg(buf),
		ArgValue(int32(-3)),
		ArgValue(uint32(7)),
		ArgValue(float32(1.5)),
		ArgValue(float64(2.5)),
	)
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if p.Len() != 5 {
		t.Fatalf("Len = %d, want 5", p.Len())
	}

	cfg := LaunchConfig1D(1024, 256)
	cfg.SharedMemBytes = 32
	for i := 0; i < 3; i++ { // the packed args are reused across launches
		if err := fn.LaunchPacked(context.Background(), cfg, p); err != nil {
			t.Fatalf("LaunchPacked: %v", err)
		}
	}
	if l.launchCalls.Load() != 3 {
		t.Fatalf("launch calls = %d, want 3", l.launchCalls.Load())
	}
	if got := *(*cudasys.CUdeviceptr)(l.params[0]); got != 0xDEAD {
		t.Errorf("arg0 = %#x, want 0xDEAD", got)
	}
	if got := *(*int32)(l.params[1]); got != -3 {
		t.Errorf("arg1 = %d, want -3", got)
	}
	if got := *(*float64)(l.params[4]); got != 2.5 {
		t.Errorf("arg4 = %v, want 2.5", got)
	}
}

func TestPackRejects(t *testing.T) {
	ctx, _, _, buf := newLaunchFixture(t)
	_ = ctx
	if _, err := Pack(Arg(buf), nil); !errors.Is(err, ErrNilKernelArg) {
		t.Errorf("nil arg = %v, want ErrNilKernelArg", err)
	}
	_ = buf.Close()
	if _, err := Pack(Arg(buf)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("closed buffer = %v, want ErrBufferClosed", err)
	}
}

func TestLaunchPackedRejects(t *testing.T) {
	ctx, _, fn, buf := newLaunchFixture(t)
	_ = ctx
	p, err := Pack(Arg(buf))
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if err := fn.LaunchPacked(context.Background(), LaunchConfig{}, p); !errors.Is(err, ErrInvalidLaunchConfig) {
		t.Errorf("invalid cfg = %v, want ErrInvalidLaunchConfig", err)
	}
	if err := fn.LaunchPacked(context.Background(), LaunchConfig1D(256, 256), nil); !errors.Is(err, ErrNilKernelArg) {
		t.Errorf("nil packed = %v, want ErrNilKernelArg", err)
	}
	var nilFn *Function
	if err := nilFn.LaunchPacked(context.Background(), LaunchConfig1D(256, 256), p); !errors.Is(err, ErrNilFunction) {
		t.Errorf("nil function = %v, want ErrNilFunction", err)
	}
}

func BenchmarkLaunchPacked(b *testing.B) {
	var l launchFake
	drv := l.driver(b)
	// Override with a launch that captures nothing, so the benchmark measures
	// the gocudrv launch path rather than the fake's per-call param copy.
	drv.CuLaunchKernel = func(cudasys.CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, cudasys.CUstream, *unsafe.Pointer, *unsafe.Pointer) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	resetDriver()
	mu.Lock()
	driver = drv
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
	p, _ := Pack(Arg(buf), ArgValue(int32(1)), ArgValue(uint32(2)), ArgValue(float32(3)))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := fn.LaunchPacked(context.Background(), cfg, p); err != nil {
			b.Fatal(err)
		}
	}
}
