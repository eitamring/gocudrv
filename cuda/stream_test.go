package cuda

import (
	"context"
	"errors"
	"math"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/eitamring/gocudrv/cudasys"
)

type streamCalls struct {
	create         atomic.Int32
	createPriority atomic.Int32
	destroy        atomic.Int32
	synchronize    atomic.Int32
	lastFlags      atomic.Uint32
	lastPriority   atomic.Int32
	lastStream     atomic.Uintptr
}

func fakeStreamDriver(c *streamCalls, failDestroy *atomic.Bool) *cudasys.Driver {
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
		CuStreamCreate: func(stream *cudasys.CUstream, flags uint32) cudasys.CUresult {
			c.create.Add(1)
			c.lastFlags.Store(flags)
			*stream = 0x5151
			return cudasys.CUDA_SUCCESS
		},
		CuStreamCreateWithPriority: func(stream *cudasys.CUstream, flags uint32, priority int32) cudasys.CUresult {
			c.createPriority.Add(1)
			c.lastFlags.Store(flags)
			c.lastPriority.Store(priority)
			*stream = 0x6161
			return cudasys.CUDA_SUCCESS
		},
		CuStreamDestroy: func(stream cudasys.CUstream) cudasys.CUresult {
			c.destroy.Add(1)
			c.lastStream.Store(uintptr(stream))
			if failDestroy != nil && failDestroy.Load() {
				failDestroy.Store(false)
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}
			return cudasys.CUDA_SUCCESS
		},
		CuStreamSynchronize: func(stream cudasys.CUstream) cudasys.CUresult {
			c.synchronize.Add(1)
			c.lastStream.Store(uintptr(stream))
			return cudasys.CUDA_SUCCESS
		},
	}
}

func TestNewStreamHappy(t *testing.T) {
	var calls streamCalls
	ctx := newTestContext(t, fakeStreamDriver(&calls, nil))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	if stream == nil {
		t.Fatal("nil stream")
	}
	if calls.create.Load() != 1 {
		t.Errorf("create calls = %d, want 1", calls.create.Load())
	}
	if calls.lastFlags.Load() != streamNonBlocking {
		t.Errorf("flags = %d, want %d", calls.lastFlags.Load(), streamNonBlocking)
	}
}

func TestNewStreamWithPriority(t *testing.T) {
	var calls streamCalls
	ctx := newTestContext(t, fakeStreamDriver(&calls, nil))
	stream, err := ctx.NewStream(WithStreamPriority(-1))
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	if calls.create.Load() != 0 {
		t.Errorf("plain create calls = %d, want 0", calls.create.Load())
	}
	if calls.createPriority.Load() != 1 {
		t.Errorf("priority create calls = %d, want 1", calls.createPriority.Load())
	}
	if calls.lastFlags.Load() != streamNonBlocking {
		t.Errorf("flags = %d, want %d", calls.lastFlags.Load(), streamNonBlocking)
	}
	if calls.lastPriority.Load() != -1 {
		t.Errorf("priority = %d, want -1", calls.lastPriority.Load())
	}
}

func TestNewStreamRejects(t *testing.T) {
	var calls streamCalls
	ctx := newTestContext(t, fakeStreamDriver(&calls, nil))
	if err := ctx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var nilCtx *Context
	if _, err := nilCtx.NewStream(); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil context err = %v, want ErrNilContext", err)
	}
	if _, err := ctx.NewStream(); !errors.Is(err, ErrContextClosed) {
		t.Errorf("closed context err = %v, want ErrContextClosed", err)
	}
	if strconv.IntSize > 32 {
		priorityCtx := newTestContext(t, fakeStreamDriver(&calls, nil))
		if _, err := priorityCtx.NewStream(WithStreamPriority(int(math.MaxInt32) + 1)); !errors.Is(err, ErrInvalidStreamPriority) {
			t.Errorf("invalid priority err = %v, want ErrInvalidStreamPriority", err)
		}
	}
}

func TestStreamSynchronize(t *testing.T) {
	var calls streamCalls
	ctx := newTestContext(t, fakeStreamDriver(&calls, nil))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if calls.synchronize.Load() != 1 {
		t.Errorf("sync calls = %d, want 1", calls.synchronize.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}

	var nilStream *Stream
	if err := nilStream.Synchronize(context.Background()); !errors.Is(err, ErrNilStream) {
		t.Errorf("nil stream err = %v, want ErrNilStream", err)
	}
}

func TestStreamSynchronizeCanceledBeforeSubmit(t *testing.T) {
	var calls streamCalls
	ctx := newTestContext(t, fakeStreamDriver(&calls, nil))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	waitCtx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := stream.Synchronize(waitCtx); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if calls.synchronize.Load() != 0 {
		t.Errorf("sync calls = %d, want 0", calls.synchronize.Load())
	}
}

func TestStreamCloseIdempotentAndRetryable(t *testing.T) {
	var calls streamCalls
	var failDestroy atomic.Bool
	failDestroy.Store(true)
	ctx := newTestContext(t, fakeStreamDriver(&calls, &failDestroy))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	if err := stream.Close(); !errors.Is(err, ErrInvalidHandle) {
		t.Errorf("first Close err = %v, want ErrInvalidHandle", err)
	}
	if err := stream.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Errorf("third Close: %v", err)
	}
	if calls.destroy.Load() != 2 {
		t.Errorf("destroy calls = %d, want 2", calls.destroy.Load())
	}
	if err := stream.Synchronize(context.Background()); !errors.Is(err, ErrStreamClosed) {
		t.Errorf("sync after close err = %v, want ErrStreamClosed", err)
	}
	var nilStream *Stream
	if err := nilStream.Close(); !errors.Is(err, ErrNilStream) {
		t.Errorf("nil stream close err = %v, want ErrNilStream", err)
	}
}

func TestStreamCloseHoldsLockDuringSynchronize(t *testing.T) {
	syncEntered := make(chan struct{})
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
		CuStreamCreate: func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
			*stream = 0x5151
			return cudasys.CUDA_SUCCESS
		},
		CuStreamDestroy: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamSynchronize: func(cudasys.CUstream) cudasys.CUresult {
			close(syncEntered)
			<-mayFinish
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })
	stream, _ := ctx.NewStream()

	syncDone := make(chan error, 1)
	go func() { syncDone <- stream.Synchronize(context.Background()) }()
	select {
	case <-syncEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("synchronize did not enter driver")
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- stream.Close() }()
	select {
	case err := <-closeDone:
		t.Fatalf("close returned during synchronize: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	close(mayFinish)
	if err := <-syncDone; err != nil {
		t.Errorf("Synchronize: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Errorf("Close: %v", err)
	}
}
