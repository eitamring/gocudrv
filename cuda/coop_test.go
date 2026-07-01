package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

type coopFake struct {
	coopCalls  atomic.Int32
	plainCalls atomic.Int32
	params     []unsafe.Pointer
	gotStream  cudasys.CUstream
	gotGrid    [3]uint32
	gotBlock   [3]uint32
	gotShared  uint32
}

func (c *coopFake) driver(t testing.TB) *cudasys.Driver {
	t.Helper()
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet:      func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult { *dev = 0; return cudasys.CUDA_SUCCESS },
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSetCurrent:           func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamCreate:            func(s *cudasys.CUstream, _ uint32) cudasys.CUresult { *s = 0x5151; return cudasys.CUDA_SUCCESS },
		CuStreamDestroy:           func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemAlloc:                func(p *cudasys.CUdeviceptr, _ uint64) cudasys.CUresult { *p = 0xDEAD; return cudasys.CUDA_SUCCESS },
		CuMemFree:                 func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleLoadData:          func(mod *cudasys.CUmodule, _ *byte) cudasys.CUresult { *mod = 0xBEEF; return cudasys.CUDA_SUCCESS },
		CuModuleUnload:            func(cudasys.CUmodule) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuModuleGetFunction: func(fn *cudasys.CUfunction, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
			*fn = 0xCAFE
			return cudasys.CUDA_SUCCESS
		},
		CuLaunchKernel: func(_ cudasys.CUfunction, _, _, _, _, _, _, _ uint32, _ cudasys.CUstream, _ *unsafe.Pointer, _ *unsafe.Pointer) cudasys.CUresult {
			c.plainCalls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuLaunchCooperativeKernel: func(fn cudasys.CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, shared uint32, stream cudasys.CUstream, params *unsafe.Pointer) cudasys.CUresult {
			c.coopCalls.Add(1)
			c.gotStream = stream
			c.gotGrid = [3]uint32{gridX, gridY, gridZ}
			c.gotBlock = [3]uint32{blockX, blockY, blockZ}
			c.gotShared = shared
			if fn != 0xCAFE {
				t.Errorf("fn = %#x, want 0xCAFE", fn)
			}
			c.params = append([]unsafe.Pointer(nil), unsafe.Slice(params, 2)...)
			return cudasys.CUDA_SUCCESS
		},
	}
}

func newCoopFixture(t *testing.T) (*Context, *Function, *Buffer[float32], *coopFake) {
	t.Helper()
	c := &coopFake{}
	installDriver(t, c.driver(t))
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
	return ctx, fn, buf, c
}

func TestLaunchCooperativeDispatches(t *testing.T) {
	_, fn, buf, c := newCoopFixture(t)
	cfg := LaunchConfig1D(1024, 256)
	cfg.SharedMemBytes = 32
	if err := fn.LaunchCooperative(context.Background(), cfg, Arg(buf), ArgValue(int32(4))); err != nil {
		t.Fatalf("LaunchCooperative: %v", err)
	}
	if c.coopCalls.Load() != 1 {
		t.Errorf("cooperative launches = %d, want 1", c.coopCalls.Load())
	}
	if c.plainCalls.Load() != 0 {
		t.Errorf("plain launches = %d, want 0", c.plainCalls.Load())
	}
	if c.gotStream != defaultStream {
		t.Errorf("stream = %#x, want default %#x", c.gotStream, defaultStream)
	}
	if c.gotGrid != [3]uint32{4, 1, 1} || c.gotBlock != [3]uint32{256, 1, 1} || c.gotShared != 32 {
		t.Errorf("cfg = grid %v block %v shared %d, want (4,1,1)/(256,1,1)/32", c.gotGrid, c.gotBlock, c.gotShared)
	}
	if len(c.params) != 2 {
		t.Errorf("params = %d, want 2", len(c.params))
	}
}

func TestLaunchCooperativeOnUsesStream(t *testing.T) {
	ctx, fn, buf, c := newCoopFixture(t)
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	if err := fn.LaunchCooperativeOn(context.Background(), stream, LaunchConfig1D(1024, 256), Arg(buf), ArgValue(int32(4))); err != nil {
		t.Fatalf("LaunchCooperativeOn: %v", err)
	}
	if c.coopCalls.Load() != 1 {
		t.Errorf("cooperative launches = %d, want 1", c.coopCalls.Load())
	}
	if c.gotStream != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", c.gotStream)
	}
}

func TestLaunchCooperativeSymbolUnavailable(t *testing.T) {
	ctx, fn, buf, _ := newCoopFixture(t)
	ctx.driver.CuLaunchCooperativeKernel = nil
	if err := fn.LaunchCooperative(context.Background(), LaunchConfig1D(1, 1), Arg(buf)); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("LaunchCooperative = %v, want ErrSymbolUnavailable", err)
	}
}

func TestLaunchCooperativeRejects(t *testing.T) {
	ctx, fn, buf, c := newCoopFixture(t)
	var nilFn *Function
	otherCtx, otherFn, otherBuf, _ := newCoopFixture(t)
	_ = otherFn
	closedStream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	_ = closedStream.Close()

	cases := []struct {
		name string
		call func() error
		want error
	}{
		{"nil function", func() error { return nilFn.LaunchCooperative(context.Background(), LaunchConfig1D(1, 1), Arg(buf)) }, ErrNilFunction},
		{"invalid config", func() error { return fn.LaunchCooperative(context.Background(), LaunchConfig{}) }, ErrInvalidLaunchConfig},
		{"nil arg", func() error { return fn.LaunchCooperative(context.Background(), LaunchConfig1D(1, 1), nil) }, ErrNilKernelArg},
		{"wrong context", func() error { return fn.LaunchCooperative(context.Background(), LaunchConfig1D(1, 1), Arg(otherBuf)) }, ErrContextMismatch},
		{"nil stream", func() error { return fn.LaunchCooperativeOn(context.Background(), nil, LaunchConfig1D(1, 1), Arg(buf)) }, ErrNilStream},
		{"closed stream", func() error { return fn.LaunchCooperativeOn(context.Background(), closedStream, LaunchConfig1D(1, 1)) }, ErrStreamClosed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.call(); !errors.Is(err, tc.want) {
				t.Errorf("%s = %v, want %v", tc.name, err, tc.want)
			}
		})
	}
	if c.coopCalls.Load() != 0 {
		t.Errorf("cooperative launches on rejected calls = %d, want 0", c.coopCalls.Load())
	}
	_ = otherCtx
}

func TestMaxCooperativeGridBlocks(t *testing.T) {
	ctx, fn, _, _ := newCoopFixture(t)
	ctx.driver.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(n *int32, _ cudasys.CUfunction, _ int32, _ uint64) cudasys.CUresult {
		*n = 6
		return cudasys.CUDA_SUCCESS
	}
	ctx.driver.CuDeviceGetAttribute = func(v *int32, attr int32, _ cudasys.CUdevice) cudasys.CUresult {
		if attr != int32(DeviceAttributeMultiprocessorCount) {
			t.Errorf("attr = %d, want MultiprocessorCount %d", attr, DeviceAttributeMultiprocessorCount)
		}
		*v = 8
		return cudasys.CUDA_SUCCESS
	}
	n, err := fn.MaxCooperativeGridBlocks(256, 1024)
	if err != nil {
		t.Fatalf("MaxCooperativeGridBlocks: %v", err)
	}
	if n != 48 {
		t.Errorf("blocks = %d, want 48 (6 per SM * 8 SMs)", n)
	}
}

func TestMaxCooperativeGridBlocksRejects(t *testing.T) {
	ctx, fn, _, _ := newCoopFixture(t)
	var nilFn *Function
	if _, err := nilFn.MaxCooperativeGridBlocks(256, 0); !errors.Is(err, ErrNilFunction) {
		t.Errorf("nil function = %v, want ErrNilFunction", err)
	}
	if _, err := fn.MaxCooperativeGridBlocks(0, 0); !errors.Is(err, ErrInvalidBlockSize) {
		t.Errorf("zero block size = %v, want ErrInvalidBlockSize", err)
	}
	ctx.driver.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(*int32, cudasys.CUfunction, int32, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if _, err := fn.MaxCooperativeGridBlocks(256, 0); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("occupancy error = %v, want ErrInvalidValue", err)
	}
}
