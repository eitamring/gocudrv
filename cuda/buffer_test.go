package cuda

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

type memCalls struct {
	alloc    atomic.Int32
	free     atomic.Int32
	htod     atomic.Int32
	dtoh     atomic.Int32
	lastPtr  atomic.Uintptr
	lastSize atomic.Uint64
}

func fakeMemoryDriver(c *memCalls, basePtr uint64) *cudasys.Driver {
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
		CuMemAlloc: func(p *cudasys.CUdeviceptr, b uint64) cudasys.CUresult {
			c.alloc.Add(1)
			c.lastSize.Store(b)
			*p = cudasys.CUdeviceptr(basePtr)
			return cudasys.CUDA_SUCCESS
		},
		CuMemFree: func(p cudasys.CUdeviceptr) cudasys.CUresult {
			c.free.Add(1)
			c.lastPtr.Store(uintptr(p))
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyHtoD: func(_ cudasys.CUdeviceptr, _ *byte, _ uint64) cudasys.CUresult {
			c.htod.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyDtoH: func(_ *byte, _ cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			c.dtoh.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	}
}

func newTestContext(t *testing.T, d *cudasys.Driver) *Context {
	t.Helper()
	installDriver(t, d)
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

func TestAllocHappy(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0xDEAD0000))

	buf, err := Alloc[float32](ctx, 256)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	if buf.Len() != 256 {
		t.Errorf("Len = %d, want 256", buf.Len())
	}
	if buf.Bytes() != 256*4 {
		t.Errorf("Bytes = %d, want %d", buf.Bytes(), 256*4)
	}
	if calls.lastSize.Load() != 256*4 {
		t.Errorf("alloc size = %d, want %d", calls.lastSize.Load(), 256*4)
	}
}

func TestAllocByteSizes(t *testing.T) {
	cases := []struct {
		name      string
		elemBytes uint64
		alloc     func(*Context, int) (uint64, error)
	}{
		{"int8", 1, func(c *Context, n int) (uint64, error) {
			b, e := Alloc[int8](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"int16", 2, func(c *Context, n int) (uint64, error) {
			b, e := Alloc[int16](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"int32", 4, func(c *Context, n int) (uint64, error) {
			b, e := Alloc[int32](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"int64", 8, func(c *Context, n int) (uint64, error) {
			b, e := Alloc[int64](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"float32", 4, func(c *Context, n int) (uint64, error) {
			b, e := Alloc[float32](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"float64", 8, func(c *Context, n int) (uint64, error) {
			b, e := Alloc[float64](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var calls memCalls
			ctx := newTestContext(t, fakeMemoryDriver(&calls, 0))
			got, err := tc.alloc(ctx, 10)
			if err != nil {
				t.Fatalf("alloc: %v", err)
			}
			if got != 10*tc.elemBytes {
				t.Errorf("Bytes = %d, want %d", got, 10*tc.elemBytes)
			}
		})
	}
}

func TestAllocRejects(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0))

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil context",
			func() error { _, e := Alloc[float32](nil, 16); return e },
			ErrNilContext,
		},
		{
			"zero length",
			func() error { _, e := Alloc[float32](ctx, 0); return e },
			ErrInvalidLength,
		},
		{
			"negative length",
			func() error { _, e := Alloc[float32](ctx, -1); return e },
			ErrInvalidLength,
		},
		{
			"overflow",
			func() error { _, e := Alloc[float64](ctx, math.MaxInt); return e },
			ErrInvalidLength,
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

func TestAllocOnClosedContext(t *testing.T) {
	var calls memCalls
	installDriver(t, fakeMemoryDriver(&calls, 0xDEAD0000))
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := Alloc[float32](ctx, 16); !errors.Is(err, ErrContextClosed) {
		t.Errorf("err = %v, want ErrContextClosed", err)
	}
}

func TestAllocPropagatesDriverError(t *testing.T) {
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
		CuMemAlloc: func(*cudasys.CUdeviceptr, uint64) cudasys.CUresult {
			return cudasys.CUDA_ERROR_OUT_OF_MEMORY
		},
	})
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	if _, err := Alloc[float32](ctx, 16); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestBufferCloseIdempotent(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0xDEAD0000))

	buf, err := Alloc[float32](ctx, 8)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := buf.Close(); err != nil {
		t.Errorf("first close: %v", err)
	}
	if err := buf.Close(); err != nil {
		t.Errorf("second close: %v", err)
	}
	if calls.free.Load() != 1 {
		t.Errorf("free calls = %d, want 1", calls.free.Load())
	}
}

func TestBufferCloseFailureCanRetry(t *testing.T) {
	var calls memCalls
	failFirst := atomic.Bool{}
	failFirst.Store(true)
	ctx := newTestContext(t, &cudasys.Driver{
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
		CuMemFree: func(cudasys.CUdeviceptr) cudasys.CUresult {
			calls.free.Add(1)
			if failFirst.Swap(false) {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}
			return cudasys.CUDA_SUCCESS
		},
	})

	buf, err := Alloc[float32](ctx, 8)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := buf.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("first close err = %v, want ErrInvalidValue", err)
	}
	if err := buf.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if calls.free.Load() != 2 {
		t.Errorf("free calls = %d, want 2", calls.free.Load())
	}
}

func TestNilBufferMethods(t *testing.T) {
	var b *Buffer[float32]
	if got := b.Len(); got != 0 {
		t.Errorf("Len = %d, want 0", got)
	}
	if got := b.Bytes(); got != 0 {
		t.Errorf("Bytes = %d, want 0", got)
	}
	if err := b.Close(); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("Close err = %v, want ErrNilBuffer", err)
	}
	if err := b.CopyFrom(context.Background(), []float32{1}); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("CopyFrom err = %v, want ErrNilBuffer", err)
	}
	if err := b.CopyTo(context.Background(), []float32{1}); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("CopyTo err = %v, want ErrNilBuffer", err)
	}
	if err := CopyHtoD(context.Background(), b, []float32{1}); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("CopyHtoD wrapper err = %v, want ErrNilBuffer", err)
	}
	if err := CopyDtoH(context.Background(), []float32{1}, b); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("CopyDtoH wrapper err = %v, want ErrNilBuffer", err)
	}
}

func TestClosedBufferOperations(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0xDEAD0000))

	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := buf.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	src := []float32{1, 2, 3, 4}
	dst := make([]float32, 4)
	if err := buf.CopyFrom(context.Background(), src); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyFrom err = %v, want ErrBufferClosed", err)
	}
	if err := buf.CopyTo(context.Background(), dst); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyTo err = %v, want ErrBufferClosed", err)
	}
}

func TestCopyLengthMismatch(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0xDEAD0000))

	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	cases := []struct {
		name string
		fn   func() error
	}{
		{"CopyFrom shorter", func() error { return buf.CopyFrom(context.Background(), []float32{1, 2}) }},
		{"CopyFrom longer", func() error { return buf.CopyFrom(context.Background(), []float32{1, 2, 3, 4, 5}) }},
		{"CopyFrom empty", func() error { return buf.CopyFrom(context.Background(), nil) }},
		{"CopyTo shorter", func() error { return buf.CopyTo(context.Background(), make([]float32, 2)) }},
		{"CopyTo longer", func() error { return buf.CopyTo(context.Background(), make([]float32, 5)) }},
		{"CopyTo empty", func() error { return buf.CopyTo(context.Background(), nil) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, ErrLengthMismatch) {
				t.Errorf("err = %v, want ErrLengthMismatch", err)
			}
		})
	}
}

func TestCopyFromHappy(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	captured := make([]byte, len(src)*4)

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
		CuMemcpyHtoD: func(_ cudasys.CUdeviceptr, s *byte, b uint64) cudasys.CUresult {
			copy(captured, unsafe.Slice(s, b))
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	buf, err := Alloc[float32](cctx, len(src))
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	if err := buf.CopyFrom(context.Background(), src); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	wantBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)
	for i := range captured {
		if captured[i] != wantBytes[i] {
			t.Errorf("byte[%d] = %d, want %d", i, captured[i], wantBytes[i])
			break
		}
	}
}

func TestCopyToHappy(t *testing.T) {
	want := []float32{1.5, 2.5, 3.5, 4.5}
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
		CuMemcpyDtoH: func(d *byte, _ cudasys.CUdeviceptr, b uint64) cudasys.CUresult {
			wantBytes := unsafe.Slice((*byte)(unsafe.Pointer(&want[0])), len(want)*4)
			copy(unsafe.Slice(d, b), wantBytes)
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	buf, err := Alloc[float32](cctx, len(want))
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	got := make([]float32, len(want))
	if err := buf.CopyTo(context.Background(), got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestCopyCanceledBeforeSubmit(t *testing.T) {
	calls := atomic.Int32{}
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
		CuMemcpyHtoD: func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
			calls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	buf, err := Alloc[float32](cctx, 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = buf.CopyFrom(ctx, []float32{1, 2, 3, 4})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if calls.Load() != 0 {
		t.Errorf("CuMemcpyHtoD calls = %d, want 0 (pre-submit cancel)", calls.Load())
	}
}

func TestCopyCanceledAfterSubmitWaitsForCompletion(t *testing.T) {
	// The CUDA call sleeps; ctx times out before it returns. Once submitted,
	// CopyFrom still waits for completion so the caller cannot reuse src while
	// CUDA is reading it.
	calls := atomic.Int32{}
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
		CuMemcpyHtoD: func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
			time.Sleep(50 * time.Millisecond)
			calls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	buf, err := Alloc[float32](cctx, 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()

	start := time.Now()
	if err := buf.CopyFrom(ctx, []float32{1, 2, 3, 4}); err != nil {
		t.Errorf("CopyFrom err = %v, want nil", err)
	}
	if elapsed := time.Since(start); elapsed < 50*time.Millisecond {
		t.Errorf("CopyFrom returned before submitted copy completed: %v", elapsed)
	}
	if calls.Load() != 1 {
		t.Errorf("memcpy calls = %d, want 1", calls.Load())
	}
}

func TestCloseWaitsForInFlightCopy(t *testing.T) {
	copyStarted := make(chan struct{})
	copyFinish := make(chan struct{})
	freeCalls := atomic.Int32{}
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
		CuMemFree: func(cudasys.CUdeviceptr) cudasys.CUresult {
			freeCalls.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyHtoD: func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
			close(copyStarted)
			<-copyFinish
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	buf, err := Alloc[float32](cctx, 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}

	copyDone := make(chan error, 1)
	go func() { copyDone <- buf.CopyFrom(context.Background(), []float32{1, 2, 3, 4}) }()
	<-copyStarted

	closeDone := make(chan error, 1)
	go func() { closeDone <- buf.Close() }()

	select {
	case err := <-closeDone:
		t.Fatalf("Close returned before copy completed: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	if freeCalls.Load() != 0 {
		t.Fatal("cuMemFree ran while copy was still in flight")
	}

	close(copyFinish)
	if err := <-copyDone; err != nil {
		t.Fatalf("copy: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Fatalf("close: %v", err)
	}
	if freeCalls.Load() != 1 {
		t.Errorf("free calls = %d, want 1", freeCalls.Load())
	}
}

func TestCopyWrapperFunctions(t *testing.T) {
	// Quick smoke check that CopyHtoD/CopyDtoH delegate to the methods.
	src := []float32{1, 2, 3}
	captured := make([]byte, len(src)*4)
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
		CuMemcpyHtoD: func(_ cudasys.CUdeviceptr, s *byte, b uint64) cudasys.CUresult {
			copy(captured, unsafe.Slice(s, b))
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyDtoH: func(d *byte, _ cudasys.CUdeviceptr, b uint64) cudasys.CUresult {
			copy(unsafe.Slice(d, b), captured)
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	buf, err := Alloc[float32](cctx, len(src))
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	if err := CopyHtoD(context.Background(), buf, src); err != nil {
		t.Fatalf("CopyHtoD: %v", err)
	}
	got := make([]float32, len(src))
	if err := CopyDtoH(context.Background(), got, buf); err != nil {
		t.Fatalf("CopyDtoH: %v", err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Errorf("round-trip[%d] = %v, want %v", i, got[i], src[i])
		}
	}
}
