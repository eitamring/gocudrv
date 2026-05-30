package cuda

import (
	"context"
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestMemInfo(t *testing.T) {
	var calls memCalls
	ctx := newTestContext(t, fakeMemoryDriver(&calls, 0xDEAD))
	free, total, err := ctx.MemInfo()
	if err != nil {
		t.Fatalf("MemInfo: %v", err)
	}
	if free != 2048 || total != 8192 {
		t.Errorf("got (free=%d total=%d), want (2048, 8192)", free, total)
	}

	var nilCtx *Context
	if _, _, err := nilCtx.MemInfo(); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil context err = %v, want ErrNilContext", err)
	}
}

func TestZeroHappy(t *testing.T) {
	var calls memCalls
	ctx, _, buf, _ := newAsyncCopyFixture(t, &calls)
	_ = ctx
	if err := buf.Zero(context.Background()); err != nil {
		t.Fatalf("Zero: %v", err)
	}
	if calls.memset.Load() != 1 {
		t.Errorf("memset calls = %d, want 1", calls.memset.Load())
	}
	if calls.lastSize.Load() != buf.Bytes() {
		t.Errorf("memset size = %d, want %d", calls.lastSize.Load(), buf.Bytes())
	}
}

func TestZeroAsyncHappy(t *testing.T) {
	var calls memCalls
	_, stream, buf, _ := newAsyncCopyFixture(t, &calls)
	if err := buf.ZeroAsync(context.Background(), stream); err != nil {
		t.Fatalf("ZeroAsync: %v", err)
	}
	if calls.memsetAsync.Load() != 1 {
		t.Errorf("async memset calls = %d, want 1", calls.memsetAsync.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}
}

func TestCopyToDeviceHappy(t *testing.T) {
	var calls memCalls
	ctx, _, src, _ := newAsyncCopyFixture(t, &calls)
	dst, err := Alloc[float32](ctx, src.Len())
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })
	if err := src.CopyToDevice(context.Background(), dst); err != nil {
		t.Fatalf("CopyToDevice: %v", err)
	}
	if calls.dtod.Load() != 1 {
		t.Errorf("dtod calls = %d, want 1", calls.dtod.Load())
	}
}

func TestCopyToDeviceAsyncHappy(t *testing.T) {
	var calls memCalls
	ctx, stream, src, _ := newAsyncCopyFixture(t, &calls)
	dst, err := Alloc[float32](ctx, src.Len())
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })
	if err := src.CopyToDeviceAsync(context.Background(), stream, dst); err != nil {
		t.Fatalf("CopyToDeviceAsync: %v", err)
	}
	if calls.dtodAsync.Load() != 1 {
		t.Errorf("dtod async calls = %d, want 1", calls.dtodAsync.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}
}

func TestPrimitivesReject(t *testing.T) {
	var calls memCalls
	ctx, stream, buf, _ := newAsyncCopyFixture(t, &calls)
	otherCtx, otherStream, _, _ := newAsyncCopyFixture(t, &memCalls{})
	_ = otherCtx

	closedBuf, err := Alloc[float32](ctx, buf.Len())
	if err != nil {
		t.Fatalf("Alloc closedBuf: %v", err)
	}
	if err := closedBuf.Close(); err != nil {
		t.Fatalf("close closedBuf: %v", err)
	}
	closedStream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream closedStream: %v", err)
	}
	if err := closedStream.Close(); err != nil {
		t.Fatalf("close closedStream: %v", err)
	}
	shortDst, err := Alloc[float32](ctx, buf.Len()+1)
	if err != nil {
		t.Fatalf("Alloc shortDst: %v", err)
	}
	t.Cleanup(func() { _ = shortDst.Close() })
	otherCtxDst, err := Alloc[float32](otherCtx, buf.Len())
	if err != nil {
		t.Fatalf("Alloc otherCtxDst: %v", err)
	}
	t.Cleanup(func() { _ = otherCtxDst.Close() })

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"zero nil buffer", func() error {
			var b *Buffer[float32]
			return b.Zero(context.Background())
		}, ErrNilBuffer},
		{"zero closed buffer", func() error { return closedBuf.Zero(context.Background()) }, ErrBufferClosed},
		{"zeroasync nil stream", func() error { return buf.ZeroAsync(context.Background(), nil) }, ErrNilStream},
		{"zeroasync closed stream", func() error { return buf.ZeroAsync(context.Background(), closedStream) }, ErrStreamClosed},
		{"zeroasync wrong stream context", func() error { return buf.ZeroAsync(context.Background(), otherStream) }, ErrContextMismatch},
		{"copytodevice nil dst", func() error { return buf.CopyToDevice(context.Background(), nil) }, ErrNilBuffer},
		{"copytodevice closed dst", func() error { return buf.CopyToDevice(context.Background(), closedBuf) }, ErrBufferClosed},
		{"copytodevice length mismatch", func() error { return buf.CopyToDevice(context.Background(), shortDst) }, ErrLengthMismatch},
		{"copytodevice wrong context", func() error { return buf.CopyToDevice(context.Background(), otherCtxDst) }, ErrContextMismatch},
		{"copytodeviceasync nil stream", func() error { return buf.CopyToDeviceAsync(context.Background(), nil, buf) }, ErrNilStream},
		{"copytodeviceasync wrong stream context", func() error { return buf.CopyToDeviceAsync(context.Background(), otherStream, buf) }, ErrContextMismatch},
		{"copytodeviceasync length mismatch", func() error { return buf.CopyToDeviceAsync(context.Background(), stream, shortDst) }, ErrLengthMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestZeroCanceledBeforeSubmit(t *testing.T) {
	var calls memCalls
	_, _, buf, _ := newAsyncCopyFixture(t, &calls)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := buf.Zero(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if calls.memset.Load() != 0 {
		t.Errorf("memset calls = %d, want 0", calls.memset.Load())
	}
}

func TestCopyToDevicePropagatesDriverError(t *testing.T) {
	var calls memCalls
	ctx, _, src, _ := newAsyncCopyFixture(t, &calls)
	dst, err := Alloc[float32](ctx, src.Len())
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })
	src.ctx.driver.CuMemcpyDtoD = func(cudasys.CUdeviceptr, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if err := src.CopyToDevice(context.Background(), dst); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}
