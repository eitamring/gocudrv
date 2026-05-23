package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/eitamring/gocudrv/cudasys"
)

type eventCalls struct {
	create      atomic.Int32
	destroy     atomic.Int32
	record      atomic.Int32
	sync        atomic.Int32
	wait        atomic.Int32
	elapsed     atomic.Int32
	lastFlags   atomic.Uint32
	lastEvent   atomic.Uintptr
	lastStream  atomic.Uintptr
	lastWaitFor atomic.Uintptr
}

func fakeEventDriver(c *eventCalls, failDestroy *atomic.Bool) *cudasys.Driver {
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
		CuStreamCreate: func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
			*stream = 0x5151
			return cudasys.CUDA_SUCCESS
		},
		CuStreamDestroy:     func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamSynchronize: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamWaitEvent: func(stream cudasys.CUstream, event cudasys.CUevent, flags uint32) cudasys.CUresult {
			c.wait.Add(1)
			c.lastStream.Store(uintptr(stream))
			c.lastWaitFor.Store(uintptr(event))
			c.lastFlags.Store(flags)
			return cudasys.CUDA_SUCCESS
		},
		CuEventCreate: func(event *cudasys.CUevent, flags uint32) cudasys.CUresult {
			c.create.Add(1)
			c.lastFlags.Store(flags)
			*event = cudasys.CUevent(0xE700 + c.create.Load())
			return cudasys.CUDA_SUCCESS
		},
		CuEventDestroy: func(event cudasys.CUevent) cudasys.CUresult {
			c.destroy.Add(1)
			c.lastEvent.Store(uintptr(event))
			if failDestroy != nil && failDestroy.Load() {
				failDestroy.Store(false)
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}
			return cudasys.CUDA_SUCCESS
		},
		CuEventRecord: func(event cudasys.CUevent, stream cudasys.CUstream) cudasys.CUresult {
			c.record.Add(1)
			c.lastEvent.Store(uintptr(event))
			c.lastStream.Store(uintptr(stream))
			return cudasys.CUDA_SUCCESS
		},
		CuEventQuery: func(event cudasys.CUevent) cudasys.CUresult {
			c.lastEvent.Store(uintptr(event))
			return cudasys.CUDA_SUCCESS
		},
		CuEventSynchronize: func(event cudasys.CUevent) cudasys.CUresult {
			c.sync.Add(1)
			c.lastEvent.Store(uintptr(event))
			return cudasys.CUDA_SUCCESS
		},
		CuEventElapsedTime: func(ms *float32, start, end cudasys.CUevent) cudasys.CUresult {
			c.elapsed.Add(1)
			c.lastEvent.Store(uintptr(start))
			c.lastWaitFor.Store(uintptr(end))
			*ms = 1.25
			return cudasys.CUDA_SUCCESS
		},
	}
}

func newEventFixture(t *testing.T, calls *eventCalls) (*Context, *Stream, *Event) {
	t.Helper()
	ctx := newTestContext(t, fakeEventDriver(calls, nil))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	event, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	t.Cleanup(func() { _ = event.Close() })
	return ctx, stream, event
}

func TestNewEventHappy(t *testing.T) {
	var calls eventCalls
	ctx := newTestContext(t, fakeEventDriver(&calls, nil))
	event, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	t.Cleanup(func() { _ = event.Close() })
	if event == nil {
		t.Fatal("nil event")
	}
	if calls.create.Load() != 1 {
		t.Errorf("create calls = %d, want 1", calls.create.Load())
	}
	if calls.lastFlags.Load() != eventDefault {
		t.Errorf("flags = %d, want %d", calls.lastFlags.Load(), eventDefault)
	}
}

func TestNewEventOptions(t *testing.T) {
	var calls eventCalls
	ctx := newTestContext(t, fakeEventDriver(&calls, nil))
	event, err := ctx.NewEvent(WithEventBlockingSync(), WithEventDisableTiming())
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	t.Cleanup(func() { _ = event.Close() })
	want := eventBlockingSync | eventDisableTiming
	if calls.lastFlags.Load() != want {
		t.Errorf("flags = %d, want %d", calls.lastFlags.Load(), want)
	}
	if !event.timingDisabled {
		t.Error("timingDisabled = false, want true")
	}
}

func TestNewEventRejects(t *testing.T) {
	var calls eventCalls
	ctx := newTestContext(t, fakeEventDriver(&calls, nil))
	if err := ctx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var nilCtx *Context
	if _, err := nilCtx.NewEvent(); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil context err = %v, want ErrNilContext", err)
	}
	if _, err := ctx.NewEvent(); !errors.Is(err, ErrContextClosed) {
		t.Errorf("closed context err = %v, want ErrContextClosed", err)
	}
}

func TestNewEventStopsApplyingOptionsAfterError(t *testing.T) {
	var calls eventCalls
	ctx := newTestContext(t, fakeEventDriver(&calls, nil))
	ran := false
	_, err := ctx.NewEvent(
		eventOptionFunc(func(opts *eventOptions) { opts.err = ErrInvalidValue }),
		eventOptionFunc(func(*eventOptions) { ran = true }),
	)
	if !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("err = %v, want ErrInvalidValue", err)
	}
	if ran {
		t.Error("later option ran after an earlier option failed")
	}
	if calls.create.Load() != 0 {
		t.Errorf("event creation ran after option failure: %d", calls.create.Load())
	}
}

func TestWaitEventStopsApplyingOptionsAfterError(t *testing.T) {
	var calls eventCalls
	_, stream, event := newEventFixture(t, &calls)
	ran := false
	err := stream.WaitEvent(
		event,
		waitOptionFunc(func(opts *waitOptions) { opts.err = ErrInvalidValue }),
		waitOptionFunc(func(*waitOptions) { ran = true }),
	)
	if !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("err = %v, want ErrInvalidValue", err)
	}
	if ran {
		t.Error("later option ran after an earlier option failed")
	}
	if calls.wait.Load() != 0 {
		t.Errorf("wait ran after option failure: %d", calls.wait.Load())
	}
}

func TestEventRecordAndStreamWaitEvent(t *testing.T) {
	var calls eventCalls
	_, stream, event := newEventFixture(t, &calls)
	if err := event.Record(stream); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if calls.record.Load() != 1 {
		t.Errorf("record calls = %d, want 1", calls.record.Load())
	}
	if calls.lastEvent.Load() != 0xE701 {
		t.Errorf("record event = %#x, want 0xE701", calls.lastEvent.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("record stream = %#x, want 0x5151", calls.lastStream.Load())
	}

	if err := stream.WaitEvent(event); err != nil {
		t.Fatalf("WaitEvent: %v", err)
	}
	if calls.wait.Load() != 1 {
		t.Errorf("wait calls = %d, want 1", calls.wait.Load())
	}
	if calls.lastWaitFor.Load() != 0xE701 {
		t.Errorf("wait event = %#x, want 0xE701", calls.lastWaitFor.Load())
	}
	if calls.lastFlags.Load() != 0 {
		t.Errorf("wait flags = %d, want 0", calls.lastFlags.Load())
	}
}

func TestEventAndWaitRejects(t *testing.T) {
	var calls eventCalls
	ctx, stream, event := newEventFixture(t, &calls)
	_, otherStream, otherEvent := newEventFixture(t, &eventCalls{})

	closedEvent, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent closedEvent: %v", err)
	}
	if err := closedEvent.Close(); err != nil {
		t.Fatalf("close closedEvent: %v", err)
	}
	closedStream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream closedStream: %v", err)
	}
	if err := closedStream.Close(); err != nil {
		t.Fatalf("close closedStream: %v", err)
	}

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil event record", func() error {
			var e *Event
			return e.Record(stream)
		}, ErrNilEvent},
		{"nil stream record", func() error { return event.Record(nil) }, ErrNilStream},
		{"closed event record", func() error { return closedEvent.Record(stream) }, ErrEventClosed},
		{"closed stream record", func() error { return event.Record(closedStream) }, ErrStreamClosed},
		{"wrong stream context record", func() error { return event.Record(otherStream) }, ErrContextMismatch},
		{"nil stream wait", func() error {
			var s *Stream
			return s.WaitEvent(event)
		}, ErrNilStream},
		{"nil event wait", func() error { return stream.WaitEvent(nil) }, ErrNilEvent},
		{"closed stream wait", func() error { return closedStream.WaitEvent(event) }, ErrStreamClosed},
		{"closed event wait", func() error { return stream.WaitEvent(closedEvent) }, ErrEventClosed},
		{"wrong event context wait", func() error { return stream.WaitEvent(otherEvent) }, ErrContextMismatch},
		{"driver error record", func() error {
			event.ctx.driver.CuEventRecord = func(cudasys.CUevent, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}
			return event.Record(stream)
		}, ErrInvalidHandle},
		{"driver error wait", func() error {
			stream.ctx.driver.CuStreamWaitEvent = func(cudasys.CUstream, cudasys.CUevent, uint32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}
			return stream.WaitEvent(event)
		}, ErrInvalidHandle},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestEventSynchronize(t *testing.T) {
	var calls eventCalls
	_, _, event := newEventFixture(t, &calls)
	if err := event.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if calls.sync.Load() != 1 {
		t.Errorf("sync calls = %d, want 1", calls.sync.Load())
	}
	if calls.lastEvent.Load() != 0xE701 {
		t.Errorf("event = %#x, want 0xE701", calls.lastEvent.Load())
	}

	var nilEvent *Event
	if err := nilEvent.Synchronize(context.Background()); !errors.Is(err, ErrNilEvent) {
		t.Errorf("nil event err = %v, want ErrNilEvent", err)
	}
}

func TestEventSynchronizeCanceledBeforeSubmit(t *testing.T) {
	var calls eventCalls
	_, _, event := newEventFixture(t, &calls)
	waitCtx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := event.Synchronize(waitCtx); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if calls.sync.Load() != 0 {
		t.Errorf("sync calls = %d, want 0", calls.sync.Load())
	}
}

func TestEventQuery(t *testing.T) {
	var calls eventCalls
	_, _, event := newEventFixture(t, &calls)
	if err := event.Query(); err != nil {
		t.Fatalf("Query: %v", err)
	}
	if calls.lastEvent.Load() != 0xE701 {
		t.Errorf("event = %#x, want 0xE701", calls.lastEvent.Load())
	}

	event.ctx.driver.CuEventQuery = func(cudasys.CUevent) cudasys.CUresult {
		return cudasys.CUDA_ERROR_NOT_READY
	}
	if err := event.Query(); !errors.Is(err, ErrNotReady) {
		t.Errorf("not-ready Query err = %v, want ErrNotReady", err)
	}

	var nilEvent *Event
	if err := nilEvent.Query(); !errors.Is(err, ErrNilEvent) {
		t.Errorf("nil event err = %v, want ErrNilEvent", err)
	}
}

func TestEventQueryClosed(t *testing.T) {
	var calls eventCalls
	_, _, event := newEventFixture(t, &calls)
	if err := event.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := event.Query(); !errors.Is(err, ErrEventClosed) {
		t.Errorf("closed event err = %v, want ErrEventClosed", err)
	}
}

func TestEventElapsed(t *testing.T) {
	var calls eventCalls
	ctx := newTestContext(t, fakeEventDriver(&calls, nil))
	start, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent start: %v", err)
	}
	t.Cleanup(func() { _ = start.Close() })
	end, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent end: %v", err)
	}
	t.Cleanup(func() { _ = end.Close() })

	elapsed, err := start.Elapsed(end)
	if err != nil {
		t.Fatalf("Elapsed: %v", err)
	}
	if elapsed != 1250*time.Microsecond {
		t.Errorf("elapsed = %v, want 1.25ms", elapsed)
	}
	if calls.elapsed.Load() != 1 {
		t.Errorf("elapsed calls = %d, want 1", calls.elapsed.Load())
	}
}

func TestEventElapsedRejects(t *testing.T) {
	var calls eventCalls
	ctx, _, event := newEventFixture(t, &calls)
	_, _, otherEvent := newEventFixture(t, &eventCalls{})
	closedEvent, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent closedEvent: %v", err)
	}
	if err := closedEvent.Close(); err != nil {
		t.Fatalf("close closedEvent: %v", err)
	}
	timingDisabled, err := ctx.NewEvent(WithEventDisableTiming())
	if err != nil {
		t.Fatalf("NewEvent timingDisabled: %v", err)
	}
	t.Cleanup(func() { _ = timingDisabled.Close() })

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil start", func() error {
			var e *Event
			_, err := e.Elapsed(event)
			return err
		}, ErrNilEvent},
		{"nil end", func() error {
			_, err := event.Elapsed(nil)
			return err
		}, ErrNilEvent},
		{"closed start", func() error {
			_, err := closedEvent.Elapsed(event)
			return err
		}, ErrEventClosed},
		{"closed end", func() error {
			_, err := event.Elapsed(closedEvent)
			return err
		}, ErrEventClosed},
		{"wrong context", func() error {
			_, err := event.Elapsed(otherEvent)
			return err
		}, ErrContextMismatch},
		{"timing disabled start", func() error {
			_, err := timingDisabled.Elapsed(event)
			return err
		}, ErrEventTimingDisabled},
		{"timing disabled end", func() error {
			_, err := event.Elapsed(timingDisabled)
			return err
		}, ErrEventTimingDisabled},
		{"driver error", func() error {
			event.ctx.driver.CuEventElapsedTime = func(*float32, cudasys.CUevent, cudasys.CUevent) cudasys.CUresult {
				return cudasys.CUDA_ERROR_NOT_READY
			}
			_, err := event.Elapsed(event)
			return err
		}, ErrNotReady},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestEventElapsedConcurrentOppositeOrder(t *testing.T) {
	var calls eventCalls
	ctx := newTestContext(t, fakeEventDriver(&calls, nil))
	a, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })

	done := make(chan error, 2)
	go func() {
		_, err := a.Elapsed(b)
		done <- err
	}()
	go func() {
		_, err := b.Elapsed(a)
		done <- err
	}()
	for i := 0; i < 2; i++ {
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("Elapsed: %v", err)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("Elapsed deadlocked")
		}
	}
}

func TestEventCloseIdempotentAndRetryable(t *testing.T) {
	var calls eventCalls
	var failDestroy atomic.Bool
	failDestroy.Store(true)
	ctx := newTestContext(t, fakeEventDriver(&calls, &failDestroy))
	event, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	if err := event.Close(); !errors.Is(err, ErrInvalidHandle) {
		t.Errorf("first Close err = %v, want ErrInvalidHandle", err)
	}
	if err := event.Synchronize(context.Background()); err != nil {
		t.Errorf("Synchronize after failed Close: %v", err)
	}
	if err := event.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if err := event.Close(); err != nil {
		t.Errorf("third Close: %v", err)
	}
	if calls.destroy.Load() != 2 {
		t.Errorf("destroy calls = %d, want 2", calls.destroy.Load())
	}
	if err := event.Synchronize(context.Background()); !errors.Is(err, ErrEventClosed) {
		t.Errorf("sync after close err = %v, want ErrEventClosed", err)
	}
	var nilEvent *Event
	if err := nilEvent.Close(); !errors.Is(err, ErrNilEvent) {
		t.Errorf("nil event close err = %v, want ErrNilEvent", err)
	}
}

func TestEventCloseHoldsLockDuringSynchronize(t *testing.T) {
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
		CuEventCreate: func(event *cudasys.CUevent, _ uint32) cudasys.CUresult {
			*event = 0xE701
			return cudasys.CUDA_SUCCESS
		},
		CuEventDestroy: func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuEventQuery:   func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuEventSynchronize: func(cudasys.CUevent) cudasys.CUresult {
			close(syncEntered)
			<-mayFinish
			return cudasys.CUDA_SUCCESS
		},
	})
	dev, _ := GetDevice(0)
	ctx, _ := dev.Primary()
	t.Cleanup(func() { _ = ctx.Close() })
	event, _ := ctx.NewEvent()

	syncDone := make(chan error, 1)
	go func() { syncDone <- event.Synchronize(context.Background()) }()
	select {
	case <-syncEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("synchronize did not enter driver")
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- event.Close() }()
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
