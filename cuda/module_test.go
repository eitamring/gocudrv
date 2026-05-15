package cuda

import (
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

type moduleFake struct {
	loadCalls   atomic.Int32
	unloadCalls atomic.Int32
	getFnCalls  atomic.Int32
	lastImage   []byte
	lastName    []byte
	lastModule  atomic.Uintptr
}

func (m *moduleFake) driver(failUnload *atomic.Bool) *cudasys.Driver {
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
		CuModuleLoadData: func(mod *cudasys.CUmodule, image *byte) cudasys.CUresult {
			m.loadCalls.Add(1)
			// Copy the bytes the driver sees including the null terminator.
			length := 0
			for {
				b := *(*byte)(unsafe.Add(unsafe.Pointer(image), length))
				length++
				if b == 0 {
					break
				}
			}
			m.lastImage = append([]byte(nil), unsafe.Slice(image, length)...)
			*mod = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
		CuModuleUnload: func(mod cudasys.CUmodule) cudasys.CUresult {
			m.unloadCalls.Add(1)
			m.lastModule.Store(uintptr(mod))
			if failUnload != nil && failUnload.Load() {
				failUnload.Store(false)
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}
			return cudasys.CUDA_SUCCESS
		},
		CuModuleGetFunction: func(fn *cudasys.CUfunction, _ cudasys.CUmodule, name *byte) cudasys.CUresult {
			m.getFnCalls.Add(1)
			length := 0
			for {
				b := *(*byte)(unsafe.Add(unsafe.Pointer(name), length))
				length++
				if b == 0 {
					break
				}
			}
			m.lastName = append([]byte(nil), unsafe.Slice(name, length)...)
			*fn = 0xCAFE
			return cudasys.CUDA_SUCCESS
		},
	}
}

func newModuleTestContext(t *testing.T, f *moduleFake, failUnload *atomic.Bool) *Context {
	t.Helper()
	installDriver(t, f.driver(failUnload))
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("get device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	return ctx
}

func TestLoadModuleHappy(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	image := []byte{'P', 'T', 'X', ' ', 'B', 'O', 'D', 'Y', 0}
	mod, err := ctx.LoadModule(image)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	if mod == nil {
		t.Fatal("nil module")
	}
	if f.loadCalls.Load() != 1 {
		t.Errorf("load calls = %d, want 1", f.loadCalls.Load())
	}
	if len(f.lastImage) != len(image) {
		t.Fatalf("image length = %d, want %d", len(f.lastImage), len(image))
	}
	for i := range image {
		if f.lastImage[i] != image[i] {
			t.Errorf("byte[%d] = %d, want %d", i, f.lastImage[i], image[i])
		}
	}
}

func TestLoadModuleAppendsNullTerminator(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	image := []byte{'P', 'T', 'X'}
	mod, err := ctx.LoadModule(image)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	want := []byte{'P', 'T', 'X', 0}
	if len(f.lastImage) != len(want) {
		t.Fatalf("image length = %d, want %d", len(f.lastImage), len(want))
	}
	for i := range want {
		if f.lastImage[i] != want[i] {
			t.Errorf("byte[%d] = %d, want %d", i, f.lastImage[i], want[i])
		}
	}
	// Ensure the original slice was not mutated.
	if len(image) != 3 {
		t.Errorf("caller image was resized: len = %d, want 3", len(image))
	}
}

func TestLoadModulePreservesNullTerminated(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	image := []byte{'P', 'T', 'X', 0}
	mod, err := ctx.LoadModule(image)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	if len(f.lastImage) != len(image) {
		t.Fatalf("image length = %d, want %d", len(f.lastImage), len(image))
	}
	for i := range image {
		if f.lastImage[i] != image[i] {
			t.Errorf("byte[%d] = %d, want %d", i, f.lastImage[i], image[i])
		}
	}
}

func TestLoadModuleRejects(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	closedCtxDriver := f.driver(nil)
	installClosed := func() *Context {
		// Spin up a separate context and close it.
		installDriver(t, closedCtxDriver)
		dev, _ := GetDevice(0)
		c, err := dev.Primary()
		if err != nil {
			t.Fatalf("primary: %v", err)
		}
		if err := c.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		return c
	}

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil context",
			func() error {
				var c *Context
				_, e := c.LoadModule([]byte{'P', 0})
				return e
			},
			ErrNilContext,
		},
		{
			"nil image",
			func() error {
				_, e := ctx.LoadModule(nil)
				return e
			},
			ErrEmptyImage,
		},
		{
			"empty image",
			func() error {
				_, e := ctx.LoadModule([]byte{})
				return e
			},
			ErrEmptyImage,
		},
		{
			"closed context",
			func() error {
				c := installClosed()
				_, e := c.LoadModule([]byte{'P', 0})
				return e
			},
			ErrContextClosed,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestLoadModulePropagatesDriverError(t *testing.T) {
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
		CuModuleLoadData: func(*cudasys.CUmodule, *byte) cudasys.CUresult {
			return cudasys.CUDA_ERROR_INVALID_PTX
		},
	})
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	if _, err := ctx.LoadModule([]byte{'P', 0}); !errors.Is(err, ErrInvalidPTX) {
		t.Errorf("err = %v, want ErrInvalidPTX", err)
	}
}

func TestLoadModuleFromFile(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	dir := t.TempDir()
	path := filepath.Join(dir, "module.ptx")
	contents := []byte{'P', 'T', 'X', ' ', 'B', 'O', 'D', 'Y'}
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	mod, err := ctx.LoadModuleFromFile(path)
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	want := append(append([]byte(nil), contents...), 0)
	if len(f.lastImage) != len(want) {
		t.Fatalf("image length = %d, want %d", len(f.lastImage), len(want))
	}
	for i := range want {
		if f.lastImage[i] != want[i] {
			t.Errorf("byte[%d] = %d, want %d", i, f.lastImage[i], want[i])
		}
	}
}

func TestLoadModuleFromFileRejects(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	if _, err := ctx.LoadModuleFromFile(""); !errors.Is(err, ErrEmptyImage) {
		t.Errorf("empty path: err = %v, want ErrEmptyImage", err)
	}

	missing := filepath.Join(t.TempDir(), "does-not-exist.ptx")
	_, err := ctx.LoadModuleFromFile(missing)
	if err == nil {
		t.Fatal("missing file: want error")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("missing file: err = %v, want os.ErrNotExist wrapped", err)
	}
}

func TestFunctionLookupHappy(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	mod, err := ctx.LoadModule([]byte{'P', 'T', 'X', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	if fn == nil {
		t.Fatal("nil function")
	}
	if fn.Name() != "vector_add" {
		t.Errorf("Name = %q, want vector_add", fn.Name())
	}
	if f.getFnCalls.Load() != 1 {
		t.Errorf("get function calls = %d, want 1", f.getFnCalls.Load())
	}
	wantName := []byte{'v', 'e', 'c', 't', 'o', 'r', '_', 'a', 'd', 'd', 0}
	if len(f.lastName) != len(wantName) {
		t.Fatalf("name length = %d, want %d", len(f.lastName), len(wantName))
	}
	for i := range wantName {
		if f.lastName[i] != wantName[i] {
			t.Errorf("name byte[%d] = %d, want %d", i, f.lastName[i], wantName[i])
		}
	}
}

func TestFunctionRejects(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}

	closedMod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule closed: %v", err)
	}
	if err := closedMod.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil module",
			func() error {
				var m *Module
				_, e := m.Function("k")
				return e
			},
			ErrNilModule,
		},
		{
			"empty name",
			func() error {
				_, e := mod.Function("")
				return e
			},
			ErrEmptyFunctionName,
		},
		{
			"closed module",
			func() error {
				_, e := closedMod.Function("k")
				return e
			},
			ErrModuleClosed,
		},
		{
			"name with embedded null",
			func() error {
				_, e := mod.Function("vector_add\x00wrong")
				return e
			},
			ErrInvalidFunctionName,
		},
		{
			"name with leading null",
			func() error {
				_, e := mod.Function("\x00")
				return e
			},
			ErrInvalidFunctionName,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestFunctionEmbeddedNullDoesNotReachDriver(t *testing.T) {
	// A name with an embedded null must be rejected up front; the driver
	// must never see a truncated request that could bind the wrong kernel.
	getFnCalls := atomic.Int32{}
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
		CuModuleLoadData: func(mod *cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*mod = 0xBABE
			return cudasys.CUDA_SUCCESS
		},
		CuModuleUnload: func(cudasys.CUmodule) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleGetFunction: func(fn *cudasys.CUfunction, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
			getFnCalls.Add(1)
			*fn = 0xF0
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	mod, err := cctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	if _, err := mod.Function("vector_add\x00wrong"); !errors.Is(err, ErrInvalidFunctionName) {
		t.Errorf("err = %v, want ErrInvalidFunctionName", err)
	}
	if getFnCalls.Load() != 0 {
		t.Errorf("CuModuleGetFunction calls = %d, want 0 (rejection must precede driver)", getFnCalls.Load())
	}
}

func TestFunctionPropagatesError(t *testing.T) {
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
		CuModuleLoadData: func(mod *cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*mod = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
		CuModuleUnload: func(cudasys.CUmodule) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleGetFunction: func(*cudasys.CUfunction, cudasys.CUmodule, *byte) cudasys.CUresult {
			return cudasys.CUDA_ERROR_NOT_FOUND
		},
	})
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	if _, err := mod.Function("missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestModuleCloseIdempotent(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	if err := mod.Close(); err != nil {
		t.Errorf("first close: %v", err)
	}
	if err := mod.Close(); err != nil {
		t.Errorf("second close: %v", err)
	}
	if f.unloadCalls.Load() != 1 {
		t.Errorf("unload calls = %d, want 1", f.unloadCalls.Load())
	}
}

func TestModuleCloseFailureCanRetry(t *testing.T) {
	var f moduleFake
	var fail atomic.Bool
	fail.Store(true)
	ctx := newModuleTestContext(t, &f, &fail)

	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	if err := mod.Close(); err == nil {
		t.Fatal("first close: want error")
	}
	if err := mod.Close(); err != nil {
		t.Errorf("retry close: %v", err)
	}
	if f.unloadCalls.Load() != 2 {
		t.Errorf("unload calls = %d, want 2", f.unloadCalls.Load())
	}
}

func TestModuleFunctionAfterCloseReturnsClosed(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)

	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	if err := mod.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := mod.Function("k"); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("err = %v, want ErrModuleClosed", err)
	}
}

func TestFunctionNameNilReceiver(t *testing.T) {
	var f *Function
	if got := f.Name(); got != "" {
		t.Errorf("Name = %q, want empty", got)
	}
}

func TestModuleCloseHoldsLockDuringUnload(t *testing.T) {
	unloadEntered := make(chan struct{})
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
		CuModuleLoadData: func(mod *cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*mod = 0xBEEF
			return cudasys.CUDA_SUCCESS
		},
		CuModuleUnload: func(cudasys.CUmodule) cudasys.CUresult {
			close(unloadEntered)
			<-mayFinish
			return cudasys.CUDA_SUCCESS
		},
		CuModuleGetFunction: func(fn *cudasys.CUfunction, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*fn = 0xCAFE
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- mod.Close() }()

	// Wait until Close has entered the driver call, so the Module's
	// exclusive lock is held.
	select {
	case <-unloadEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("unload did not enter the driver call")
	}

	fnDone := make(chan error, 1)
	go func() {
		_, e := mod.Function("k")
		fnDone <- e
	}()

	// Function takes the RLock, so it must block behind Close's exclusive
	// Lock and not return while Unload is still running.
	select {
	case e := <-fnDone:
		t.Fatalf("Function returned before Close finished: %v", e)
	case <-time.After(20 * time.Millisecond):
		// good: Function is blocked behind the module write lock
	}

	close(mayFinish)
	if err := <-closeDone; err != nil {
		t.Errorf("close: %v", err)
	}
	// After Close, Function must observe the closed state.
	if err := <-fnDone; !errors.Is(err, ErrModuleClosed) {
		t.Errorf("Function err = %v, want ErrModuleClosed", err)
	}
}
