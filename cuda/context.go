package cuda

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/executor"
)

// Context wraps a CUDA primary context plus a pinned command executor. Blocking
// copies and synchronization lazily create their own pinned executors so
// ordinary commands can continue while the caller waits for CUDA.
type Context struct {
	device     *Device
	driver     *cudasys.Driver
	raw        cudasys.CUcontext
	exec       *executor.Executor
	copyMu     sync.Mutex
	copyExec   *executor.Executor
	syncMu     sync.Mutex
	syncExec   *executor.Executor
	syncActive atomic.Int64
	commandErr error
	opMu       sync.RWMutex
	closed     atomic.Bool
}

// Primary retains the primary context on the device and binds it as the
// current context on a dedicated pinned command thread. The returned Context
// owns that executor and any lazily created copy or synchronization executor;
// call Close to release the context and stop them.
//
// On failure all partial state (retained context, started executor) is
// rolled back before returning.
func (d *Device) Primary() (*Context, error) {
	drv := currentDriver()
	if drv == nil {
		return nil, ErrNotInitialized
	}
	if d == nil {
		return nil, ErrNilDevice
	}

	exec := executor.New()

	var raw cudasys.CUcontext
	err := exec.Do(func() error {
		c, e := cudaresult.PrimaryCtxRetain(drv, d.handle)
		if e != nil {
			return e
		}
		if e := cudaresult.CtxSetCurrent(drv, c); e != nil {
			return errors.Join(e, cudaresult.PrimaryCtxRelease(drv, d.handle))
		}
		raw = c
		return nil
	})
	if err != nil {
		_ = exec.Close()
		return nil, err
	}

	return &Context{
		device: d,
		driver: drv,
		raw:    raw,
		exec:   exec,
	}, nil
}

// Device returns the device this context was created on.
func (c *Context) Device() *Device {
	if c == nil {
		return nil
	}
	return c.device
}

// Synchronize blocks until all preceding work in the context has finished
// or ctx is canceled. Canceling ctx stops the wait; the GPU work continues
// regardless. Pass context.Background() if no cancellation is needed.
func (c *Context) Synchronize(ctx context.Context) error {
	if c == nil {
		return ErrNilContext
	}
	return c.doJobCtx(ctx, newSyncOp(c.driver, opCtxSync, 0, 0, 0))
}

// StreamPriorityRange returns the least and greatest meaningful stream
// priorities for this context. Lower numeric values represent higher priority;
// the meaningful CUDA interval is [greatest, least]. Devices without priority
// support return (0, 0).
func (c *Context) StreamPriorityRange() (least, greatest int, err error) {
	if c == nil {
		return 0, 0, ErrNilContext
	}
	err = c.do(context.Background(), func() error {
		l, g, e := cudaresult.CtxGetStreamPriorityRange(c.driver)
		if e != nil {
			return e
		}
		least = int(l)
		greatest = int(g)
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	return least, greatest, nil
}

// MemInfo returns the free and total device memory in bytes for this context's
// device. The free value reflects the whole device, not just this context.
func (c *Context) MemInfo() (free, total uint64, err error) {
	if c == nil {
		return 0, 0, ErrNilContext
	}
	err = c.do(context.Background(), func() error {
		f, t, e := cudaresult.MemGetInfo(c.driver)
		if e != nil {
			return e
		}
		free = f
		total = t
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	return free, total, nil
}

// do runs fn on the context's pinned command executor with cancellation.
func (c *Context) do(ctx context.Context, fn func() error) error {
	return c.doWith(ctx, fn, false, false)
}

// doWait waits for fn after submission without draining synchronization work.
func (c *Context) doWait(ctx context.Context, fn func() error) error {
	return c.doWith(ctx, fn, true, false)
}

// doBarrier drains synchronization work before running a teardown command.
func (c *Context) doBarrier(ctx context.Context, fn func() error) error {
	return c.doWith(ctx, fn, true, true)
}

func (c *Context) newExecutor() (*executor.Executor, error) {
	candidate := executor.New()
	if err := candidate.Do(func() error {
		return cudaresult.CtxSetCurrent(c.driver, c.raw)
	}); err != nil {
		candidate.Retire()
		_ = candidate.Close()
		return nil, err
	}
	return candidate, nil
}

func (c *Context) copyExecutor() (*executor.Executor, error) {
	c.copyMu.Lock()
	defer c.copyMu.Unlock()
	if c.copyExec != nil {
		return c.copyExec, nil
	}

	candidate, err := c.newExecutor()
	if err != nil {
		return nil, err
	}
	c.copyExec = candidate
	return candidate, nil
}

func (c *Context) syncExecutor() (*executor.Executor, error) {
	c.syncMu.Lock()
	defer c.syncMu.Unlock()
	if c.syncExec != nil {
		return c.syncExec, nil
	}

	candidate, err := c.newExecutor()
	if err != nil {
		return nil, err
	}
	c.syncExec = candidate
	return candidate, nil
}

func (c *Context) syncBarrier(ctx context.Context) error {
	if c.syncActive.Load() == 0 {
		return nil
	}
	c.syncMu.Lock()
	lane := c.syncExec
	c.syncMu.Unlock()
	if lane == nil {
		return nil
	}
	return lane.DoCtx(ctx, syncBarrierNoop)
}

func syncBarrierNoop() error { return nil }

type trackedSyncJob struct {
	ctx *Context
	job executor.Job
}

func (j *trackedSyncJob) Run() error {
	return j.job.Run()
}

func (j *trackedSyncJob) Recycle() {
	ctx, job := j.ctx, j.job
	*j = trackedSyncJob{}
	executor.RecycleJob(job)
	ctx.syncActive.Add(-1)
	trackedSyncJobPool.Put(j)
}

var trackedSyncJobPool = sync.Pool{New: func() any { return new(trackedSyncJob) }}

// doJob runs a pooled Job on the command executor with the same wait semantics
// as doWait, but without allocating a closure per call. Used by the hot copy,
// memset, and launch paths.
func (c *Context) doJob(ctx context.Context, j executor.Job) error {
	if c == nil {
		executor.RecycleJob(j)
		return ErrNilContext
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	if c.exec == nil {
		executor.RecycleJob(j)
		return ErrNilContext
	}
	if c.closed.Load() {
		executor.RecycleJob(j)
		return ErrContextClosed
	}
	if c.commandErr != nil {
		if err := ctx.Err(); err != nil {
			executor.RecycleJob(j)
			return err
		}
		executor.RecycleJob(j)
		return c.commandErr
	}
	return c.exec.DoJob(ctx, j)
}

// doCopyJob runs a caller-recycled copy job on the pinned copy executor. A
// canceled context can stop submission, but an accepted copy always finishes
// before this method returns so Go memory remains valid for the driver call.
func (c *Context) doCopyJob(ctx context.Context, j executor.Job) error {
	if c == nil {
		return ErrNilContext
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	if c.exec == nil {
		return ErrNilContext
	}
	if c.closed.Load() {
		return ErrContextClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.commandErr != nil {
		return c.commandErr
	}
	lane, err := c.copyExecutor()
	if err != nil {
		return err
	}
	return lane.DoJob(ctx, j)
}

func (c *Context) doCopyWait(ctx context.Context, fn func() error) error {
	if c == nil {
		return ErrNilContext
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	if c.exec == nil {
		return ErrNilContext
	}
	if c.closed.Load() {
		return ErrContextClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.commandErr != nil {
		return c.commandErr
	}
	lane, err := c.copyExecutor()
	if err != nil {
		return err
	}
	return lane.DoCtxWait(ctx, fn)
}

// doJobCtx runs a pooled Job on the synchronization executor. The job must be
// executor-recycled (implement Recycle), never pooled by the caller.
func (c *Context) doJobCtx(ctx context.Context, j executor.Job) error {
	if c == nil {
		executor.RecycleJob(j)
		return ErrNilContext
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	if c.exec == nil {
		executor.RecycleJob(j)
		return ErrNilContext
	}
	if c.closed.Load() {
		executor.RecycleJob(j)
		return ErrContextClosed
	}
	if err := ctx.Err(); err != nil {
		executor.RecycleJob(j)
		return err
	}
	if c.commandErr != nil {
		executor.RecycleJob(j)
		return c.commandErr
	}
	lane, err := c.syncExecutor()
	if err != nil {
		executor.RecycleJob(j)
		return err
	}
	tracked := trackedSyncJobPool.Get().(*trackedSyncJob)
	tracked.ctx, tracked.job = c, j
	c.syncActive.Add(1)
	return lane.DoJobCtx(ctx, tracked)
}

func (c *Context) doWith(ctx context.Context, fn func() error, waitAfterSubmit, barrierBeforeSubmit bool) error {
	if c == nil {
		return ErrNilContext
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	if c.exec == nil {
		return ErrNilContext
	}
	if c.closed.Load() {
		return ErrContextClosed
	}
	if c.commandErr != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
		return c.commandErr
	}
	if waitAfterSubmit {
		if err := ctx.Err(); err != nil {
			return err
		}
		if barrierBeforeSubmit {
			if err := c.syncBarrier(ctx); err != nil {
				return err
			}
		}
		return c.exec.DoCtxWait(ctx, fn)
	}
	return c.exec.DoCtx(ctx, fn)
}

func (c *Context) closeSyncExecutor() error {
	c.syncMu.Lock()
	lane := c.syncExec
	c.syncExec = nil
	c.syncMu.Unlock()
	if lane == nil {
		return nil
	}

	return c.closeExecutor(lane)
}

func (c *Context) closeCopyExecutor() error {
	c.copyMu.Lock()
	lane := c.copyExec
	c.copyExec = nil
	c.copyMu.Unlock()
	if lane == nil {
		return nil
	}
	return c.closeExecutor(lane)
}

func (c *Context) closeExecutor(lane *executor.Executor) error {
	clearErr := lane.Do(func() error {
		return cudaresult.CtxSetCurrent(c.driver, 0)
	})
	if clearErr != nil {
		lane.Retire()
	}
	return errors.Join(clearErr, lane.Close())
}

func (c *Context) recoverCommandExecutor() error {
	if c.commandErr == nil {
		return nil
	}
	candidate := executor.New()
	if err := candidate.Do(func() error {
		return cudaresult.CtxSetCurrent(c.driver, c.raw)
	}); err != nil {
		candidate.Retire()
		_ = candidate.Close()
		c.commandErr = err
		return err
	}
	c.exec = candidate
	c.commandErr = nil
	return nil
}

// Close releases the primary context retain and stops every started executor.
// After a successful Close, all Context methods return ErrContextClosed and
// further Close calls return nil. If releasing the primary context fails the
// retain count was not dropped, so the Context stays open for Close to be
// retried.
func (c *Context) Close() error {
	if c == nil || c.device == nil {
		return ErrNilContext
	}
	c.opMu.Lock()
	defer c.opMu.Unlock()
	if c.exec == nil {
		return ErrNilContext
	}
	if c.closed.Load() {
		return nil
	}
	if err := c.recoverCommandExecutor(); err != nil {
		return err
	}

	syncErr := c.closeSyncExecutor()
	copyErr := c.closeCopyExecutor()
	var clearErr, releaseErr, restoreErr error
	if err := c.exec.Do(func() error {
		clearErr = cudaresult.CtxSetCurrent(c.driver, 0)
		releaseErr = cudaresult.PrimaryCtxRelease(c.driver, c.device.handle)
		if releaseErr != nil {
			restoreErr = cudaresult.CtxSetCurrent(c.driver, c.raw)
		}
		return nil
	}); err != nil {
		return errors.Join(syncErr, copyErr, err)
	}
	if releaseErr != nil {
		if restoreErr != nil {
			c.commandErr = restoreErr
			c.exec.Retire()
			_ = c.exec.Close()
		}
		return errors.Join(syncErr, copyErr, clearErr, releaseErr, restoreErr)
	}
	c.closed.Store(true)
	if clearErr != nil {
		c.exec.Retire()
	}
	return errors.Join(syncErr, copyErr, clearErr, c.exec.Close())
}
