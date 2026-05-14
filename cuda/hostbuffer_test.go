package cuda

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/eitamring/gocudrv/cudasys"
)

type hostMemFake struct {
	storage [][]byte
	alloc   atomic.Int32
	free    atomic.Int32
}

func (f *hostMemFake) driver(failFree *atomic.Bool) *cudasys.Driver {
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
		CuMemAllocHost: func(pp **byte, b uint64) cudasys.CUresult {
			f.alloc.Add(1)
			buf := make([]byte, b)
			f.storage = append(f.storage, buf)
			*pp = &buf[0]
			return cudasys.CUDA_SUCCESS
		},
		CuMemFreeHost: func(*byte) cudasys.CUresult {
			f.free.Add(1)
			if failFree != nil && failFree.Load() {
				failFree.Store(false)
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}
			return cudasys.CUDA_SUCCESS
		},
		CuMemAlloc: func(p *cudasys.CUdeviceptr, _ uint64) cudasys.CUresult {
			*p = 0xDEAD
			return cudasys.CUDA_SUCCESS
		},
		CuMemFree:    func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemcpyHtoD: func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemcpyDtoH: func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
	}
}

func newHostTestContext(t *testing.T, f *hostMemFake, fail *atomic.Bool) *Context {
	t.Helper()
	installDriver(t, f.driver(fail))
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

func TestAllocHostHappy(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	h, err := AllocHost[float32](ctx, 256)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })
	if h.Len() != 256 {
		t.Errorf("Len = %d, want 256", h.Len())
	}
	if h.Bytes() != 256*4 {
		t.Errorf("Bytes = %d, want %d", h.Bytes(), 256*4)
	}
	if f.alloc.Load() != 1 {
		t.Errorf("alloc calls = %d, want 1", f.alloc.Load())
	}
}

func TestAllocHostByteSizes(t *testing.T) {
	cases := []struct {
		name      string
		elemBytes uint64
		alloc     func(*Context, int) (uint64, error)
	}{
		{"int8", 1, func(c *Context, n int) (uint64, error) {
			b, e := AllocHost[int8](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"int16", 2, func(c *Context, n int) (uint64, error) {
			b, e := AllocHost[int16](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"int32", 4, func(c *Context, n int) (uint64, error) {
			b, e := AllocHost[int32](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"int64", 8, func(c *Context, n int) (uint64, error) {
			b, e := AllocHost[int64](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"float32", 4, func(c *Context, n int) (uint64, error) {
			b, e := AllocHost[float32](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
		{"float64", 8, func(c *Context, n int) (uint64, error) {
			b, e := AllocHost[float64](c, n)
			if e != nil {
				return 0, e
			}
			return b.Bytes(), nil
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var f hostMemFake
			var fail atomic.Bool
			ctx := newHostTestContext(t, &f, &fail)
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

func TestAllocHostRejects(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil context",
			func() error { _, e := AllocHost[float32](nil, 16); return e },
			ErrNilContext,
		},
		{
			"zero length",
			func() error { _, e := AllocHost[float32](ctx, 0); return e },
			ErrInvalidLength,
		},
		{
			"negative length",
			func() error { _, e := AllocHost[float32](ctx, -1); return e },
			ErrInvalidLength,
		},
		{
			"overflow",
			func() error { _, e := AllocHost[float64](ctx, math.MaxInt); return e },
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

func TestAllocHostOnClosedContext(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	installDriver(t, f.driver(&fail))
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := AllocHost[float32](ctx, 16); !errors.Is(err, ErrContextClosed) {
		t.Errorf("err = %v, want ErrContextClosed", err)
	}
}

func TestAllocHostPropagatesDriverError(t *testing.T) {
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
		CuMemAllocHost: func(**byte, uint64) cudasys.CUresult {
			return cudasys.CUDA_ERROR_OUT_OF_MEMORY
		},
	})
	dev, _ := GetDevice(0)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	if _, err := AllocHost[float32](ctx, 16); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestHostSliceLenAndWriteRead(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	h, err := AllocHost[float32](ctx, 8)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })

	s := h.Slice()
	if len(s) != 8 {
		t.Errorf("len = %d, want 8", len(s))
	}
	if cap(s) != 8 {
		t.Errorf("cap = %d, want 8", cap(s))
	}
	for i := range s {
		s[i] = float32(i) * 1.5
	}
	s2 := h.Slice()
	for i, want := range []float32{0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5} {
		if s2[i] != want {
			t.Errorf("s2[%d] = %v, want %v", i, s2[i], want)
		}
	}
}

func TestHostSliceDistinctAddresses(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	a, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })

	if &a.Slice()[0] == &b.Slice()[0] {
		t.Error("two HostBuffer allocations returned the same address")
	}
}

func TestHostSliceAfterCloseReturnsNil(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	h, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if got := h.Slice(); got != nil {
		t.Errorf("Slice after close = %v, want nil", got)
	}
}

func TestHostCloseIdempotent(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	h, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Errorf("first close: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Errorf("second close: %v", err)
	}
	if f.free.Load() != 1 {
		t.Errorf("free calls = %d, want 1", f.free.Load())
	}
}

func TestHostCloseFailureCanRetry(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	fail.Store(true)
	ctx := newHostTestContext(t, &f, &fail)

	h, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	if err := h.Close(); err == nil {
		t.Fatal("first close: want error")
	}
	// fail flipped to false inside the fake on first call; second succeeds.
	if err := h.Close(); err != nil {
		t.Errorf("retry close: %v", err)
	}
	if f.free.Load() != 2 {
		t.Errorf("free calls = %d, want 2", f.free.Load())
	}
}

func TestNilHostBufferMethods(t *testing.T) {
	var h *HostBuffer[float32]
	if got := h.Len(); got != 0 {
		t.Errorf("Len = %d, want 0", got)
	}
	if got := h.Bytes(); got != 0 {
		t.Errorf("Bytes = %d, want 0", got)
	}
	if got := h.Slice(); got != nil {
		t.Errorf("Slice = %v, want nil", got)
	}
	if err := h.Close(); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("Close err = %v, want ErrNilBuffer", err)
	}
}

func TestHostBufferFeedsDeviceCopy(t *testing.T) {
	// The safe path: Buffer.CopyFromHost / CopyToHost. Holds the host
	// buffer's RLock for the duration of the copy.
	var f hostMemFake
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
		CuMemAllocHost: func(pp **byte, b uint64) cudasys.CUresult {
			buf := make([]byte, b)
			f.storage = append(f.storage, buf)
			*pp = &buf[0]
			return cudasys.CUDA_SUCCESS
		},
		CuMemFreeHost: func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemcpyHtoD:  func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemcpyDtoH:  func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
	})

	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	host, err := AllocHost[float32](cctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = host.Close() })

	dst, err := Alloc[float32](cctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	if err := dst.CopyFromHost(context.Background(), host); err != nil {
		t.Errorf("CopyFromHost: %v", err)
	}
	if err := dst.CopyToHost(context.Background(), host); err != nil {
		t.Errorf("CopyToHost: %v", err)
	}
}

func TestCopyFromHostRejections(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	dst, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	host, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	mismatched, err := AllocHost[float32](ctx, 8)
	if err != nil {
		t.Fatalf("AllocHost mismatched: %v", err)
	}
	t.Cleanup(func() { _ = mismatched.Close() })

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil dst buffer",
			func() error {
				var b *Buffer[float32]
				return b.CopyFromHost(context.Background(), host)
			},
			ErrNilBuffer,
		},
		{
			"nil host buffer",
			func() error {
				return dst.CopyFromHost(context.Background(), nil)
			},
			ErrNilBuffer,
		},
		{
			"length mismatch",
			func() error {
				return dst.CopyFromHost(context.Background(), mismatched)
			},
			ErrLengthMismatch,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}

	// Closed host buffer: should return ErrBufferClosed rather than the
	// length-mismatch reported by the nil-slice path on CopyFrom.
	if err := host.Close(); err != nil {
		t.Fatalf("close host: %v", err)
	}
	if err := dst.CopyFromHost(context.Background(), host); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("closed host err = %v, want ErrBufferClosed", err)
	}
}

// lockTestFake builds a driver where CuMemcpyHtoD and CuMemcpyDtoH block
// until the test signals them. The fake closes copyEntered the moment a
// copy starts so the test can synchronize without timing-based sleeps.
func lockTestFake(f *hostMemFake, htoDEntered, dtoHEntered chan<- struct{}, mayFinish <-chan struct{}) *cudasys.Driver {
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
		CuMemAllocHost: func(pp **byte, b uint64) cudasys.CUresult {
			buf := make([]byte, b)
			f.storage = append(f.storage, buf)
			*pp = &buf[0]
			return cudasys.CUDA_SUCCESS
		},
		CuMemFreeHost: func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuMemcpyHtoD: func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
			if htoDEntered != nil {
				close(htoDEntered)
			}
			<-mayFinish
			return cudasys.CUDA_SUCCESS
		},
		CuMemcpyDtoH: func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
			if dtoHEntered != nil {
				close(dtoHEntered)
			}
			<-mayFinish
			return cudasys.CUDA_SUCCESS
		},
	}
}

func assertCloseBlocksOnCopy(t *testing.T, host *HostBuffer[float32], copyFn func() error, copyEntered <-chan struct{}, mayFinish chan<- struct{}) {
	t.Helper()
	copyDone := make(chan error, 1)
	go func() { copyDone <- copyFn() }()

	// Wait until the copy is observably inside the fake driver call, so
	// we know the host buffer's RLock is held.
	select {
	case <-copyEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("copy did not enter the driver call")
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- host.Close() }()

	// Close must not return while the copy is still running.
	select {
	case err := <-closeDone:
		t.Fatalf("Close returned before copy finished: %v", err)
	case <-time.After(20 * time.Millisecond):
		// good: Close is blocked behind the host RLock
	}

	close(mayFinish)
	if err := <-copyDone; err != nil {
		t.Errorf("copy: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Errorf("close: %v", err)
	}
}

func TestCopyFromHostHoldsLockDuringCopy(t *testing.T) {
	var f hostMemFake
	entered := make(chan struct{})
	mayFinish := make(chan struct{})
	installDriver(t, lockTestFake(&f, entered, nil, mayFinish))

	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	host, err := AllocHost[float32](cctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	dst, err := Alloc[float32](cctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	assertCloseBlocksOnCopy(t, host, func() error {
		return dst.CopyFromHost(context.Background(), host)
	}, entered, mayFinish)
}

func TestCopyToHostHoldsLockDuringCopy(t *testing.T) {
	var f hostMemFake
	entered := make(chan struct{})
	mayFinish := make(chan struct{})
	installDriver(t, lockTestFake(&f, nil, entered, mayFinish))

	dev, _ := GetDevice(0)
	cctx, _ := dev.Primary()
	t.Cleanup(func() { _ = cctx.Close() })

	host, err := AllocHost[float32](cctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	src, err := Alloc[float32](cctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })

	assertCloseBlocksOnCopy(t, host, func() error {
		return src.CopyToHost(context.Background(), host)
	}, entered, mayFinish)
}

func TestCopyToHostRejections(t *testing.T) {
	var f hostMemFake
	var fail atomic.Bool
	ctx := newHostTestContext(t, &f, &fail)

	src, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })

	host, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	mismatched, err := AllocHost[float32](ctx, 8)
	if err != nil {
		t.Fatalf("AllocHost mismatched: %v", err)
	}
	t.Cleanup(func() { _ = mismatched.Close() })

	cases := []struct {
		name    string
		fn      func() error
		wantErr error
	}{
		{
			"nil src buffer",
			func() error {
				var b *Buffer[float32]
				return b.CopyToHost(context.Background(), host)
			},
			ErrNilBuffer,
		},
		{
			"nil host buffer",
			func() error {
				return src.CopyToHost(context.Background(), nil)
			},
			ErrNilBuffer,
		},
		{
			"length mismatch",
			func() error {
				return src.CopyToHost(context.Background(), mismatched)
			},
			ErrLengthMismatch,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}

	if err := host.Close(); err != nil {
		t.Fatalf("close host: %v", err)
	}
	if err := src.CopyToHost(context.Background(), host); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("closed host err = %v, want ErrBufferClosed", err)
	}
}
