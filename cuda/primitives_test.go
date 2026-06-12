package cuda

import (
	"context"
	"errors"
	"math"
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

func TestFillHappy(t *testing.T) {
	var calls memCalls
	_, _, buf, _ := newAsyncCopyFixture(t, &calls)
	if err := buf.Fill(context.Background(), float32(1.5)); err != nil {
		t.Fatalf("Fill: %v", err)
	}
	if calls.memset.Load() != 1 {
		t.Errorf("memset calls = %d, want 1", calls.memset.Load())
	}
	if calls.lastSize.Load() != uint64(buf.Len()) {
		t.Errorf("memset count = %d, want %d (element count, not bytes)", calls.lastSize.Load(), buf.Len())
	}
	if calls.lastVal.Load() != uint64(math.Float32bits(1.5)) {
		t.Errorf("pattern = %#x, want %#x", calls.lastVal.Load(), math.Float32bits(1.5))
	}
}

func TestFillAsyncHappy(t *testing.T) {
	var calls memCalls
	_, stream, buf, _ := newAsyncCopyFixture(t, &calls)
	if err := buf.FillAsync(context.Background(), stream, float32(2.5)); err != nil {
		t.Fatalf("FillAsync: %v", err)
	}
	if calls.memsetAsync.Load() != 1 {
		t.Errorf("async memset calls = %d, want 1", calls.memsetAsync.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}
	if calls.lastVal.Load() != uint64(math.Float32bits(2.5)) {
		t.Errorf("pattern = %#x, want %#x", calls.lastVal.Load(), math.Float32bits(2.5))
	}
}

// TestFillWidthDispatch checks that each element width routes to the matching
// memset primitive and that the value is reinterpreted bit-for-bit.
func TestFillWidthDispatch(t *testing.T) {
	var calls memCalls
	ctx, _, _, _ := newAsyncCopyFixture(t, &calls)

	t.Run("int8 via D8", func(t *testing.T) {
		b, err := Alloc[int8](ctx, 4)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		t.Cleanup(func() { _ = b.Close() })
		if err := b.Fill(context.Background(), int8(0x7F)); err != nil {
			t.Fatalf("Fill: %v", err)
		}
		if calls.lastVal.Load() != 0x7F || calls.lastSize.Load() != 4 {
			t.Errorf("got val=%#x count=%d, want 0x7f, 4", calls.lastVal.Load(), calls.lastSize.Load())
		}
	})
	t.Run("uint16 via D16", func(t *testing.T) {
		b, err := Alloc[uint16](ctx, 4)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		t.Cleanup(func() { _ = b.Close() })
		if err := b.Fill(context.Background(), uint16(0xBEEF)); err != nil {
			t.Fatalf("Fill: %v", err)
		}
		if calls.lastVal.Load() != 0xBEEF || calls.lastSize.Load() != 4 {
			t.Errorf("got val=%#x count=%d, want 0xbeef, 4", calls.lastVal.Load(), calls.lastSize.Load())
		}
	})
	t.Run("int32 via D32", func(t *testing.T) {
		b, err := Alloc[int32](ctx, 4)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		t.Cleanup(func() { _ = b.Close() })
		if err := b.Fill(context.Background(), int32(-1)); err != nil {
			t.Fatalf("Fill: %v", err)
		}
		if calls.lastVal.Load() != uint64(uint32(0xFFFFFFFF)) {
			t.Errorf("got val=%#x, want 0xffffffff", calls.lastVal.Load())
		}
	})
}

func TestFillUnsupportedType(t *testing.T) {
	var calls memCalls
	ctx, stream, _, _ := newAsyncCopyFixture(t, &calls)
	b, err := Alloc[float64](ctx, 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	if err := b.Fill(context.Background(), float64(1)); !errors.Is(err, ErrUnsupportedFillType) {
		t.Errorf("Fill err = %v, want ErrUnsupportedFillType", err)
	}
	if err := b.FillAsync(context.Background(), stream, float64(1)); !errors.Is(err, ErrUnsupportedFillType) {
		t.Errorf("FillAsync err = %v, want ErrUnsupportedFillType", err)
	}
	if calls.memset.Load() != 0 || calls.memsetAsync.Load() != 0 {
		t.Errorf("memset ran for an unsupported type: sync=%d async=%d", calls.memset.Load(), calls.memsetAsync.Load())
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
		{"fill nil buffer", func() error {
			var b *Buffer[float32]
			return b.Fill(context.Background(), 0)
		}, ErrNilBuffer},
		{"fill closed buffer", func() error { return closedBuf.Fill(context.Background(), 0) }, ErrBufferClosed},
		{"fillasync nil stream", func() error { return buf.FillAsync(context.Background(), nil, 0) }, ErrNilStream},
		{"fillasync closed stream", func() error { return buf.FillAsync(context.Background(), closedStream, 0) }, ErrStreamClosed},
		{"fillasync wrong stream context", func() error { return buf.FillAsync(context.Background(), otherStream, 0) }, ErrContextMismatch},
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
