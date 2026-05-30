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
	alloc       atomic.Int32
	free        atomic.Int32
	htod        atomic.Int32
	dtoh        atomic.Int32
	htodAsync   atomic.Int32
	dtohAsync   atomic.Int32
	dtod        atomic.Int32
	dtodAsync   atomic.Int32
	memset      atomic.Int32
	memsetAsync atomic.Int32
	lastPtr     atomic.Uintptr
	lastSize    atomic.Uint64
	lastStream  atomic.Uintptr
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
		CuMemcpyHtoDAsync: func(_ cudasys.CUdeviceptr, _ *byte, _ uint64, stream cudasys.CUstream) cudasys.CUresult {
			c.htodAsync.Add(1)
			c.lastStream.Store(uintptr(stream))
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyDtoHAsync: func(_ *byte, _ cudasys.CUdeviceptr, _ uint64, stream cudasys.CUstream) cudasys.CUresult {
			c.dtohAsync.Add(1)
			c.lastStream.Store(uintptr(stream))
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyDtoD: func(_, _ cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			c.dtod.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyDtoDAsync: func(_, _ cudasys.CUdeviceptr, _ uint64, stream cudasys.CUstream) cudasys.CUresult {
			c.dtodAsync.Add(1)
			c.lastStream.Store(uintptr(stream))
			return cudasys.CUDA_SUCCESS
		},
		CuMemsetD8: func(_ cudasys.CUdeviceptr, _ uint8, n uint64) cudasys.CUresult {
			c.memset.Add(1)
			c.lastSize.Store(n)
			return cudasys.CUDA_SUCCESS
		},
		CuMemsetD8Async: func(_ cudasys.CUdeviceptr, _ uint8, n uint64, stream cudasys.CUstream) cudasys.CUresult {
			c.memsetAsync.Add(1)
			c.lastSize.Store(n)
			c.lastStream.Store(uintptr(stream))
			return cudasys.CUDA_SUCCESS
		},
		CuMemGetInfo: func(free, total *uint64) cudasys.CUresult {
			*free = 2048
			*total = 8192
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

func newAsyncCopyFixture(t *testing.T, calls *memCalls) (*Context, *Stream, *Buffer[float32], *HostBuffer[float32]) {
	t.Helper()
	drv := fakeMemoryDriver(calls, 0xDEAD)
	drv.CuMemAllocHost = func(pp **byte, bytes uint64) cudasys.CUresult {
		storage := make([]byte, int(bytes))
		*pp = &storage[0]
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemFreeHost = func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuStreamCreate = func(stream *cudasys.CUstream, flags uint32) cudasys.CUresult {
		if flags != streamNonBlocking {
			t.Errorf("stream flags = %d, want %d", flags, streamNonBlocking)
		}
		*stream = 0x5151
		return cudasys.CUDA_SUCCESS
	}
	drv.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuStreamSynchronize = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }

	ctx := newTestContext(t, drv)
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	host, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = host.Close() })
	return ctx, stream, buf, host
}

func TestCopyFromHostAsyncHappy(t *testing.T) {
	var calls memCalls
	var captured []byte
	_, stream, buf, host := newAsyncCopyFixture(t, &calls)
	src := host.Slice()
	for i := range src {
		src[i] = float32(i + 1)
	}
	buf.ctx.driver.CuMemcpyHtoDAsync = func(dst cudasys.CUdeviceptr, src *byte, bytes uint64, stream cudasys.CUstream) cudasys.CUresult {
		calls.htodAsync.Add(1)
		calls.lastStream.Store(uintptr(stream))
		if dst != 0xDEAD {
			t.Errorf("dst = %#x, want 0xDEAD", dst)
		}
		captured = append([]byte(nil), unsafe.Slice(src, bytes)...)
		return cudasys.CUDA_SUCCESS
	}

	if err := buf.CopyFromHostAsync(context.Background(), stream, host); err != nil {
		t.Fatalf("CopyFromHostAsync: %v", err)
	}
	if calls.htodAsync.Load() != 1 {
		t.Errorf("async htod calls = %d, want 1", calls.htodAsync.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}
	wantBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)
	for i := range wantBytes {
		if captured[i] != wantBytes[i] {
			t.Errorf("captured[%d] = %d, want %d", i, captured[i], wantBytes[i])
			break
		}
	}
}

func TestCopyToHostAsyncHappy(t *testing.T) {
	var calls memCalls
	_, stream, buf, host := newAsyncCopyFixture(t, &calls)
	want := []float32{1.5, 2.5, 3.5, 4.5}
	buf.ctx.driver.CuMemcpyDtoHAsync = func(dst *byte, src cudasys.CUdeviceptr, bytes uint64, stream cudasys.CUstream) cudasys.CUresult {
		calls.dtohAsync.Add(1)
		calls.lastStream.Store(uintptr(stream))
		if src != 0xDEAD {
			t.Errorf("src = %#x, want 0xDEAD", src)
		}
		copy(unsafe.Slice(dst, bytes), unsafe.Slice((*byte)(unsafe.Pointer(&want[0])), len(want)*4))
		return cudasys.CUDA_SUCCESS
	}

	if err := buf.CopyToHostAsync(context.Background(), stream, host); err != nil {
		t.Fatalf("CopyToHostAsync: %v", err)
	}
	if calls.dtohAsync.Load() != 1 {
		t.Errorf("async dtoh calls = %d, want 1", calls.dtohAsync.Load())
	}
	got := host.Slice()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestCopyHostAsyncRejects(t *testing.T) {
	var calls memCalls
	ctx, stream, buf, host := newAsyncCopyFixture(t, &calls)
	_, otherStream, _, otherHost := newAsyncCopyFixture(t, &memCalls{})

	closedBuf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc closedBuf: %v", err)
	}
	if err := closedBuf.Close(); err != nil {
		t.Fatalf("close closedBuf: %v", err)
	}
	closedHost, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost closedHost: %v", err)
	}
	if err := closedHost.Close(); err != nil {
		t.Fatalf("close closedHost: %v", err)
	}
	closedStream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream closedStream: %v", err)
	}
	if err := closedStream.Close(); err != nil {
		t.Fatalf("close closedStream: %v", err)
	}
	shortHost, err := AllocHost[float32](ctx, 2)
	if err != nil {
		t.Fatalf("AllocHost shortHost: %v", err)
	}
	t.Cleanup(func() { _ = shortHost.Close() })

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil buffer from", func() error {
			var b *Buffer[float32]
			return b.CopyFromHostAsync(context.Background(), stream, host)
		}, ErrNilBuffer},
		{"nil host from", func() error { return buf.CopyFromHostAsync(context.Background(), stream, nil) }, ErrNilBuffer},
		{"nil stream from", func() error { return buf.CopyFromHostAsync(context.Background(), nil, host) }, ErrNilStream},
		{"closed stream from", func() error { return buf.CopyFromHostAsync(context.Background(), closedStream, host) }, ErrStreamClosed},
		{"closed buffer from", func() error { return closedBuf.CopyFromHostAsync(context.Background(), stream, host) }, ErrBufferClosed},
		{"closed host from", func() error { return buf.CopyFromHostAsync(context.Background(), stream, closedHost) }, ErrBufferClosed},
		{"length mismatch from", func() error { return buf.CopyFromHostAsync(context.Background(), stream, shortHost) }, ErrLengthMismatch},
		{"wrong stream context from", func() error { return buf.CopyFromHostAsync(context.Background(), otherStream, host) }, ErrContextMismatch},
		{"wrong host context from", func() error { return buf.CopyFromHostAsync(context.Background(), stream, otherHost) }, ErrContextMismatch},
		{"nil buffer to", func() error {
			var b *Buffer[float32]
			return b.CopyToHostAsync(context.Background(), stream, host)
		}, ErrNilBuffer},
		{"nil host to", func() error { return buf.CopyToHostAsync(context.Background(), stream, nil) }, ErrNilBuffer},
		{"nil stream to", func() error { return buf.CopyToHostAsync(context.Background(), nil, host) }, ErrNilStream},
		{"closed stream to", func() error { return buf.CopyToHostAsync(context.Background(), closedStream, host) }, ErrStreamClosed},
		{"closed buffer to", func() error { return closedBuf.CopyToHostAsync(context.Background(), stream, host) }, ErrBufferClosed},
		{"closed host to", func() error { return buf.CopyToHostAsync(context.Background(), stream, closedHost) }, ErrBufferClosed},
		{"length mismatch to", func() error { return buf.CopyToHostAsync(context.Background(), stream, shortHost) }, ErrLengthMismatch},
		{"wrong stream context to", func() error { return buf.CopyToHostAsync(context.Background(), otherStream, host) }, ErrContextMismatch},
		{"wrong host context to", func() error { return buf.CopyToHostAsync(context.Background(), stream, otherHost) }, ErrContextMismatch},
		{"driver error from", func() error {
			buf.ctx.driver.CuMemcpyHtoDAsync = func(cudasys.CUdeviceptr, *byte, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}
			return buf.CopyFromHostAsync(context.Background(), stream, host)
		}, ErrInvalidValue},
		{"driver error to", func() error {
			buf.ctx.driver.CuMemcpyDtoHAsync = func(*byte, cudasys.CUdeviceptr, uint64, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}
			return buf.CopyToHostAsync(context.Background(), stream, host)
		}, ErrInvalidValue},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestCopyHostAsyncCanceledBeforeSubmit(t *testing.T) {
	var calls memCalls
	_, stream, buf, host := newAsyncCopyFixture(t, &calls)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := buf.CopyFromHostAsync(ctx, stream, host); !errors.Is(err, context.Canceled) {
		t.Errorf("CopyFromHostAsync err = %v, want context.Canceled", err)
	}
	if err := buf.CopyToHostAsync(ctx, stream, host); !errors.Is(err, context.Canceled) {
		t.Errorf("CopyToHostAsync err = %v, want context.Canceled", err)
	}
	if calls.htodAsync.Load() != 0 {
		t.Errorf("async htod calls = %d, want 0", calls.htodAsync.Load())
	}
	if calls.dtohAsync.Load() != 0 {
		t.Errorf("async dtoh calls = %d, want 0", calls.dtohAsync.Load())
	}
}

func TestCopyHostAsyncCanceledAfterSubmitWaitsForEnqueue(t *testing.T) {
	cases := []struct {
		name      string
		install   func(*Buffer[float32], *memCalls)
		copy      func(context.Context, *Buffer[float32], *Stream, *HostBuffer[float32]) error
		callCount func(*memCalls) int32
	}{
		{
			name: "from host",
			install: func(buf *Buffer[float32], calls *memCalls) {
				buf.ctx.driver.CuMemcpyHtoDAsync = func(cudasys.CUdeviceptr, *byte, uint64, cudasys.CUstream) cudasys.CUresult {
					time.Sleep(50 * time.Millisecond)
					calls.htodAsync.Add(1)
					return cudasys.CUDA_SUCCESS
				}
			},
			copy: func(ctx context.Context, buf *Buffer[float32], stream *Stream, host *HostBuffer[float32]) error {
				return buf.CopyFromHostAsync(ctx, stream, host)
			},
			callCount: func(calls *memCalls) int32 { return calls.htodAsync.Load() },
		},
		{
			name: "to host",
			install: func(buf *Buffer[float32], calls *memCalls) {
				buf.ctx.driver.CuMemcpyDtoHAsync = func(*byte, cudasys.CUdeviceptr, uint64, cudasys.CUstream) cudasys.CUresult {
					time.Sleep(50 * time.Millisecond)
					calls.dtohAsync.Add(1)
					return cudasys.CUDA_SUCCESS
				}
			},
			copy: func(ctx context.Context, buf *Buffer[float32], stream *Stream, host *HostBuffer[float32]) error {
				return buf.CopyToHostAsync(ctx, stream, host)
			},
			callCount: func(calls *memCalls) int32 { return calls.dtohAsync.Load() },
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var calls memCalls
			_, stream, buf, host := newAsyncCopyFixture(t, &calls)
			tc.install(buf, &calls)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
			defer cancel()

			start := time.Now()
			if err := tc.copy(ctx, buf, stream, host); err != nil {
				t.Errorf("copy err = %v, want nil", err)
			}
			if elapsed := time.Since(start); elapsed < 50*time.Millisecond {
				t.Errorf("copy returned before enqueue completed: %v", elapsed)
			}
			if got := tc.callCount(&calls); got != 1 {
				t.Errorf("async copy calls = %d, want 1", got)
			}
		})
	}
}

func TestCopyFromHostAsyncDoesNotHoldResourcesAfterEnqueue(t *testing.T) {
	var calls memCalls
	_, stream, buf, host := newAsyncCopyFixture(t, &calls)

	if err := buf.CopyFromHostAsync(context.Background(), stream, host); err != nil {
		t.Fatalf("CopyFromHostAsync: %v", err)
	}
	if err := host.Close(); err != nil {
		t.Fatalf("HostBuffer.Close after async enqueue = %v, want nil", err)
	}
	if err := buf.Close(); err != nil {
		t.Fatalf("Buffer.Close after async enqueue = %v, want nil", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Stream.Close after async enqueue = %v, want nil", err)
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
