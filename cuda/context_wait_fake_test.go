package cuda

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/eitamring/gocudrv/cudasys"
)

type waitLaneCalls struct {
	rawBinds     atomic.Int32
	zeroBinds    atomic.Int32
	releases     atomic.Int32
	contextSyncs atomic.Int32
	streamSyncs  atomic.Int32
	eventSyncs   atomic.Int32
	streamQuery  atomic.Int32
	eventQuery   atomic.Int32
	streamClose  atomic.Int32
	eventClose   atomic.Int32
	memInfo      atomic.Int32
}

func waitLaneDriver(c *waitLaneCalls) *cudasys.Driver {
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult {
			*n = 1
			return cudasys.CUDA_SUCCESS
		},
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult {
			c.releases.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSetCurrent: func(ctx cudasys.CUcontext) cudasys.CUresult {
			if ctx == 0 {
				c.zeroBinds.Add(1)
			} else {
				c.rawBinds.Add(1)
			}
			return cudasys.CUDA_SUCCESS
		},
		CuCtxSynchronize: func() cudasys.CUresult {
			c.contextSyncs.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuMemGetInfo: func(free, total *uint64) cudasys.CUresult {
			c.memInfo.Add(1)
			*free = 2048
			*total = 8192
			return cudasys.CUDA_SUCCESS
		},
		CuStreamCreate: func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
			*stream = 0x5151
			return cudasys.CUDA_SUCCESS
		},
		CuStreamDestroy: func(cudasys.CUstream) cudasys.CUresult {
			c.streamClose.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuStreamSynchronize: func(cudasys.CUstream) cudasys.CUresult {
			c.streamSyncs.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult {
			c.streamQuery.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuEventCreate: func(event *cudasys.CUevent, _ uint32) cudasys.CUresult {
			*event = 0xE701
			return cudasys.CUDA_SUCCESS
		},
		CuEventDestroy: func(cudasys.CUevent) cudasys.CUresult {
			c.eventClose.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuEventSynchronize: func(cudasys.CUevent) cudasys.CUresult {
			c.eventSyncs.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuEventQuery: func(cudasys.CUevent) cudasys.CUresult {
			c.eventQuery.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	}
}

func hasSyncExecutor(c *Context) bool {
	c.syncMu.Lock()
	defer c.syncMu.Unlock()
	return c.syncExec != nil
}

func commandFailure(c *Context) error {
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	return c.commandErr
}

func waitSignal(t *testing.T, ch <-chan struct{}, name string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %s", name)
	}
}

func waitError(t *testing.T, ch <-chan error, name string) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %s", name)
		return nil
	}
}

func assertPending(t *testing.T, ch <-chan error, name string) {
	t.Helper()
	select {
	case err := <-ch:
		t.Fatalf("%s completed early: %v", name, err)
	case <-time.After(20 * time.Millisecond):
	}
}

func TestSyncExecutorIsLazy(t *testing.T) {
	var calls waitLaneCalls
	ctx := newTestContext(t, waitLaneDriver(&calls))

	if hasSyncExecutor(ctx) {
		t.Fatal("sync executor exists before first synchronization")
	}
	free, total, err := ctx.MemInfo()
	if err != nil {
		t.Fatalf("MemInfo: %v", err)
	}
	if free != 2048 || total != 8192 {
		t.Fatalf("MemInfo = (%d, %d), want (2048, 8192)", free, total)
	}
	if hasSyncExecutor(ctx) {
		t.Fatal("ordinary command created the sync executor")
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if calls.rawBinds.Load() != 1 {
		t.Fatalf("raw context binds = %d, want 1", calls.rawBinds.Load())
	}
}

func TestPreCanceledSynchronizeDoesNotCreateExecutor(t *testing.T) {
	var calls waitLaneCalls
	ctx := newTestContext(t, waitLaneDriver(&calls))
	waitCtx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := ctx.Synchronize(waitCtx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Synchronize = %v, want context.Canceled", err)
	}
	if hasSyncExecutor(ctx) {
		t.Fatal("pre-canceled synchronization created the sync executor")
	}
	if calls.contextSyncs.Load() != 0 {
		t.Fatalf("context sync calls = %d, want 0", calls.contextSyncs.Load())
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := ctx.Synchronize(waitCtx); !errors.Is(err, ErrContextClosed) {
		t.Fatalf("closed Synchronize = %v, want ErrContextClosed", err)
	}
}

func TestDoWaitChecksCancellationBeforeBarrier(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	driver.CuCtxSynchronize = func() cudasys.CUresult {
		close(entered)
		<-release
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	syncDone := make(chan error, 1)
	go func() { syncDone <- ctx.Synchronize(context.Background()) }()
	waitSignal(t, entered, "context synchronize")

	waitCtx, cancel := context.WithCancel(context.Background())
	cancel()
	ran := atomic.Bool{}
	if err := ctx.doWait(waitCtx, func() error {
		ran.Store(true)
		return nil
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("doWait = %v, want context.Canceled", err)
	}
	if ran.Load() {
		t.Fatal("canceled doWait command ran")
	}
	unblock()
	if err := waitError(t, syncDone, "context synchronize completion"); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
}

func TestDoWaitCancellationAbandonsAcceptedBarrier(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	driver.CuCtxSynchronize = func() cudasys.CUresult {
		close(entered)
		<-release
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	syncDone := make(chan error, 1)
	go func() { syncDone <- ctx.Synchronize(context.Background()) }()
	waitSignal(t, entered, "context synchronize")

	waitCtx, cancel := context.WithCancel(context.Background())
	waitDone := make(chan error, 1)
	ran := atomic.Bool{}
	started := make(chan struct{})
	go func() {
		close(started)
		waitDone <- ctx.doWait(waitCtx, func() error {
			ran.Store(true)
			return nil
		})
	}()
	waitSignal(t, started, "doWait start")
	time.Sleep(20 * time.Millisecond)
	cancel()
	if err := waitError(t, waitDone, "canceled doWait"); !errors.Is(err, context.Canceled) {
		t.Fatalf("doWait = %v, want context.Canceled", err)
	}
	if ran.Load() {
		t.Fatal("command behind canceled barrier ran")
	}
	unblock()
	if err := waitError(t, syncDone, "context synchronize completion"); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
}

func TestDoWaitSkipsIdleSyncExecutor(t *testing.T) {
	var calls waitLaneCalls
	ctx := newTestContext(t, waitLaneDriver(&calls))
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	ctx.syncMu.Lock()
	lane := ctx.syncExec
	ctx.syncMu.Unlock()
	if lane == nil {
		t.Fatal("sync executor was not created")
	}

	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	laneDone := make(chan error, 1)
	go func() {
		laneDone <- lane.Do(func() error {
			close(entered)
			<-release
			return nil
		})
	}()
	waitSignal(t, entered, "idle sync executor task")
	commandDone := make(chan error, 1)
	go func() {
		commandDone <- ctx.doWait(context.Background(), func() error { return nil })
	}()
	if err := waitError(t, commandDone, "strict command"); err != nil {
		t.Fatalf("doWait: %v", err)
	}
	unblock()
	if err := waitError(t, laneDone, "sync executor task completion"); err != nil {
		t.Fatalf("sync executor task: %v", err)
	}
}

func TestConcurrentSynchronizeCreatesOneExecutor(t *testing.T) {
	var calls waitLaneCalls
	ctx := newTestContext(t, waitLaneDriver(&calls))
	const n = 32
	errs := make(chan error, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for range n {
		go func() {
			defer wg.Done()
			errs <- ctx.Synchronize(context.Background())
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
	}
	if calls.rawBinds.Load() != 2 {
		t.Fatalf("raw context binds = %d, want 2", calls.rawBinds.Load())
	}
	if calls.contextSyncs.Load() != n {
		t.Fatalf("context sync calls = %d, want %d", calls.contextSyncs.Load(), n)
	}
}

func TestSyncExecutorBindFailureCanRetry(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	var rawAttempts atomic.Int32
	driver.CuCtxSetCurrent = func(ctx cudasys.CUcontext) cudasys.CUresult {
		if ctx == 0 {
			calls.zeroBinds.Add(1)
			return cudasys.CUDA_SUCCESS
		}
		attempt := rawAttempts.Add(1)
		calls.rawBinds.Add(1)
		if attempt == 2 {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)

	if err := ctx.Synchronize(context.Background()); !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("first Synchronize = %v, want ErrInvalidContext", err)
	}
	if hasSyncExecutor(ctx) {
		t.Fatal("failed bind stored a sync executor")
	}
	if calls.contextSyncs.Load() != 0 {
		t.Fatalf("context sync calls = %d, want 0", calls.contextSyncs.Load())
	}
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("retry Synchronize: %v", err)
	}
	if !hasSyncExecutor(ctx) {
		t.Fatal("successful retry did not store a sync executor")
	}
	if rawAttempts.Load() != 3 {
		t.Fatalf("raw bind attempts = %d, want 3", rawAttempts.Load())
	}
}

func TestBlockedSynchronizationLeavesCommandExecutorFree(t *testing.T) {
	t.Run("context", func(t *testing.T) {
		var calls waitLaneCalls
		driver := waitLaneDriver(&calls)
		entered := make(chan struct{})
		release := make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		driver.CuCtxSynchronize = func() cudasys.CUresult {
			calls.contextSyncs.Add(1)
			close(entered)
			<-release
			return cudasys.CUDA_SUCCESS
		}
		ctx := newTestContext(t, driver)
		syncDone := make(chan error, 1)
		go func() { syncDone <- ctx.Synchronize(context.Background()) }()
		waitSignal(t, entered, "context synchronize")

		commandDone := make(chan error, 1)
		go func() {
			_, _, err := ctx.MemInfo()
			commandDone <- err
		}()
		if err := waitError(t, commandDone, "MemInfo"); err != nil {
			t.Fatalf("MemInfo: %v", err)
		}
		unblock()
		if err := waitError(t, syncDone, "context synchronize completion"); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
	})

	t.Run("stream", func(t *testing.T) {
		var calls waitLaneCalls
		driver := waitLaneDriver(&calls)
		entered := make(chan struct{})
		release := make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		driver.CuStreamSynchronize = func(cudasys.CUstream) cudasys.CUresult {
			calls.streamSyncs.Add(1)
			close(entered)
			<-release
			return cudasys.CUDA_SUCCESS
		}
		ctx := newTestContext(t, driver)
		stream, err := ctx.NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		t.Cleanup(func() { _ = stream.Close() })
		syncDone := make(chan error, 1)
		go func() { syncDone <- stream.Synchronize(context.Background()) }()
		waitSignal(t, entered, "stream synchronize")

		commandDone := make(chan error, 1)
		go func() { commandDone <- stream.Query() }()
		if err := waitError(t, commandDone, "Stream.Query"); err != nil {
			t.Fatalf("Query: %v", err)
		}
		unblock()
		if err := waitError(t, syncDone, "stream synchronize completion"); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
	})

	t.Run("event", func(t *testing.T) {
		var calls waitLaneCalls
		driver := waitLaneDriver(&calls)
		entered := make(chan struct{})
		release := make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		driver.CuEventSynchronize = func(cudasys.CUevent) cudasys.CUresult {
			calls.eventSyncs.Add(1)
			close(entered)
			<-release
			return cudasys.CUDA_SUCCESS
		}
		ctx := newTestContext(t, driver)
		event, err := ctx.NewEvent()
		if err != nil {
			t.Fatalf("NewEvent: %v", err)
		}
		t.Cleanup(func() { _ = event.Close() })
		syncDone := make(chan error, 1)
		go func() { syncDone <- event.Synchronize(context.Background()) }()
		waitSignal(t, entered, "event synchronize")

		commandDone := make(chan error, 1)
		go func() { commandDone <- event.Query() }()
		if err := waitError(t, commandDone, "Event.Query"); err != nil {
			t.Fatalf("Query: %v", err)
		}
		unblock()
		if err := waitError(t, syncDone, "event synchronize completion"); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
	})
}

func TestCanceledResourceSynchronizeDelaysClose(t *testing.T) {
	t.Run("stream", func(t *testing.T) {
		var calls waitLaneCalls
		driver := waitLaneDriver(&calls)
		entered := make(chan struct{})
		release := make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		driver.CuStreamSynchronize = func(cudasys.CUstream) cudasys.CUresult {
			calls.streamSyncs.Add(1)
			close(entered)
			<-release
			return cudasys.CUDA_SUCCESS
		}
		ctx := newTestContext(t, driver)
		stream, err := ctx.NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		t.Cleanup(func() { _ = stream.Close() })
		waitCtx, cancel := context.WithCancel(context.Background())
		syncDone := make(chan error, 1)
		go func() { syncDone <- stream.Synchronize(waitCtx) }()
		waitSignal(t, entered, "stream synchronize")
		cancel()
		if err := waitError(t, syncDone, "canceled stream synchronize"); !errors.Is(err, context.Canceled) {
			t.Fatalf("Synchronize = %v, want context.Canceled", err)
		}

		closeDone := make(chan error, 1)
		go func() { closeDone <- stream.Close() }()
		assertPending(t, closeDone, "Stream.Close")
		if calls.streamClose.Load() != 0 {
			t.Fatalf("stream destroy calls = %d, want 0", calls.streamClose.Load())
		}
		unblock()
		if err := waitError(t, closeDone, "Stream.Close completion"); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if calls.streamClose.Load() != 1 {
			t.Fatalf("stream destroy calls = %d, want 1", calls.streamClose.Load())
		}
	})

	t.Run("event", func(t *testing.T) {
		var calls waitLaneCalls
		driver := waitLaneDriver(&calls)
		entered := make(chan struct{})
		release := make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		driver.CuEventSynchronize = func(cudasys.CUevent) cudasys.CUresult {
			calls.eventSyncs.Add(1)
			close(entered)
			<-release
			return cudasys.CUDA_SUCCESS
		}
		ctx := newTestContext(t, driver)
		event, err := ctx.NewEvent()
		if err != nil {
			t.Fatalf("NewEvent: %v", err)
		}
		t.Cleanup(func() { _ = event.Close() })
		waitCtx, cancel := context.WithCancel(context.Background())
		syncDone := make(chan error, 1)
		go func() { syncDone <- event.Synchronize(waitCtx) }()
		waitSignal(t, entered, "event synchronize")
		cancel()
		if err := waitError(t, syncDone, "canceled event synchronize"); !errors.Is(err, context.Canceled) {
			t.Fatalf("Synchronize = %v, want context.Canceled", err)
		}

		closeDone := make(chan error, 1)
		go func() { closeDone <- event.Close() }()
		assertPending(t, closeDone, "Event.Close")
		if calls.eventClose.Load() != 0 {
			t.Fatalf("event destroy calls = %d, want 0", calls.eventClose.Load())
		}
		unblock()
		if err := waitError(t, closeDone, "Event.Close completion"); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if calls.eventClose.Load() != 1 {
			t.Fatalf("event destroy calls = %d, want 1", calls.eventClose.Load())
		}
	})
}

func TestCanceledSynchronizeDelaysContextRelease(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	entered := make(chan struct{})
	releaseSync := make(chan struct{})
	releaseCalled := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(releaseSync) }) }
	defer unblock()
	driver.CuCtxSynchronize = func() cudasys.CUresult {
		calls.contextSyncs.Add(1)
		close(entered)
		<-releaseSync
		return cudasys.CUDA_SUCCESS
	}
	driver.CuDevicePrimaryCtxRelease = func(cudasys.CUdevice) cudasys.CUresult {
		calls.releases.Add(1)
		close(releaseCalled)
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	waitCtx, cancel := context.WithCancel(context.Background())
	syncDone := make(chan error, 1)
	go func() { syncDone <- ctx.Synchronize(waitCtx) }()
	waitSignal(t, entered, "context synchronize")
	cancel()
	if err := waitError(t, syncDone, "canceled context synchronize"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Synchronize = %v, want context.Canceled", err)
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- ctx.Close() }()
	assertPending(t, closeDone, "Context.Close")
	select {
	case <-releaseCalled:
		t.Fatal("primary context released before synchronization drained")
	default:
	}
	unblock()
	if err := waitError(t, closeDone, "Context.Close completion"); err != nil {
		t.Fatalf("Close: %v", err)
	}
	waitSignal(t, releaseCalled, "primary context release")
}

func TestSyncExecutorUnbindsBeforeRelease(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	var unbindsAtRelease atomic.Int32
	driver.CuDevicePrimaryCtxRelease = func(cudasys.CUdevice) cudasys.CUresult {
		unbindsAtRelease.Store(calls.zeroBinds.Load())
		calls.releases.Add(1)
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if calls.zeroBinds.Load() != 2 {
		t.Fatalf("zero context binds = %d, want 2", calls.zeroBinds.Load())
	}
	if unbindsAtRelease.Load() != 2 {
		t.Fatalf("unbinds at release = %d, want 2", unbindsAtRelease.Load())
	}
}

func TestSyncExecutorUnbindFailureStillReleases(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	driver.CuCtxSetCurrent = func(ctx cudasys.CUcontext) cudasys.CUresult {
		if ctx != 0 {
			calls.rawBinds.Add(1)
			return cudasys.CUDA_SUCCESS
		}
		attempt := calls.zeroBinds.Add(1)
		if attempt == 1 {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if err := ctx.Close(); !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("Close = %v, want ErrInvalidContext", err)
	}
	if calls.releases.Load() != 1 {
		t.Fatalf("release calls = %d, want 1", calls.releases.Load())
	}
	if !ctx.closed.Load() {
		t.Fatal("context left open after successful release")
	}
}

func TestReleaseFailureRestoresContext(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	driver.CuDevicePrimaryCtxRelease = func(cudasys.CUdevice) cudasys.CUresult {
		attempt := calls.releases.Add(1)
		if attempt == 1 {
			return cudasys.CUDA_ERROR_INVALID_DEVICE
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	if err := ctx.Close(); !errors.Is(err, ErrInvalidDevice) {
		t.Fatalf("first Close = %v, want ErrInvalidDevice", err)
	}
	if ctx.closed.Load() {
		t.Fatal("context closed after failed release")
	}
	if commandFailure(ctx) != nil {
		t.Fatalf("command executor quarantined after successful restore: %v", commandFailure(ctx))
	}
	if hasSyncExecutor(ctx) {
		t.Fatal("sync executor retained after failed Close")
	}
	if _, _, err := ctx.MemInfo(); err != nil {
		t.Fatalf("MemInfo after failed Close: %v", err)
	}
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize after failed Close: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
	if calls.releases.Load() != 2 {
		t.Fatalf("release calls = %d, want 2", calls.releases.Load())
	}
	if calls.rawBinds.Load() != 4 {
		t.Fatalf("raw context binds = %d, want 4", calls.rawBinds.Load())
	}
}

func TestRestoreFailureQuarantinesUntilCloseRetry(t *testing.T) {
	var calls waitLaneCalls
	driver := waitLaneDriver(&calls)
	driver.CuDevicePrimaryCtxRelease = func(cudasys.CUdevice) cudasys.CUresult {
		attempt := calls.releases.Add(1)
		if attempt == 1 {
			return cudasys.CUDA_ERROR_INVALID_DEVICE
		}
		return cudasys.CUDA_SUCCESS
	}
	driver.CuCtxSetCurrent = func(ctx cudasys.CUcontext) cudasys.CUresult {
		if ctx == 0 {
			calls.zeroBinds.Add(1)
			return cudasys.CUDA_SUCCESS
		}
		attempt := calls.rawBinds.Add(1)
		if attempt == 3 {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	if err := ctx.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	err := ctx.Close()
	if !errors.Is(err, ErrInvalidDevice) || !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("first Close = %v, want release and restore errors", err)
	}
	if ctx.closed.Load() {
		t.Fatal("context closed after failed release")
	}
	if !errors.Is(commandFailure(ctx), ErrInvalidContext) {
		t.Fatalf("command quarantine = %v, want ErrInvalidContext", commandFailure(ctx))
	}
	if _, _, err := ctx.MemInfo(); !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("MemInfo while quarantined = %v, want ErrInvalidContext", err)
	}
	if calls.memInfo.Load() != 0 {
		t.Fatalf("MemInfo driver calls = %d, want 0", calls.memInfo.Load())
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
	if calls.releases.Load() != 2 {
		t.Fatalf("release calls = %d, want 2", calls.releases.Load())
	}
	if calls.rawBinds.Load() != 4 {
		t.Fatalf("raw context binds = %d, want 4", calls.rawBinds.Load())
	}
}
