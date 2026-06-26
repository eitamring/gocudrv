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

// Context wraps a CUDA primary context plus the pinned-thread executor that
// keeps it current. Every driver call that needs context affinity routes
// through the executor.
type Context struct {
	device *Device
	driver *cudasys.Driver
	raw    cudasys.CUcontext
	exec   *executor.Executor
	opMu   sync.RWMutex
	closed atomic.Bool
}

// Primary retains the primary context on the device and binds it as the
// current context on a dedicated pinned thread. The returned Context owns
// the executor goroutine; call Close to release the context and stop it.
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
	return c.do(ctx, func() error {
		return cudaresult.CtxSynchronize(c.driver)
	})
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

// do runs fn on the context's executor with cancellation. Internal entry
// point for future memory, module, stream, and launch code so every CUDA
// call that needs context affinity routes through the same pinned thread.
func (c *Context) do(ctx context.Context, fn func() error) error {
	return c.doWith(ctx, fn, false)
}

func (c *Context) doWait(ctx context.Context, fn func() error) error {
	return c.doWith(ctx, fn, true)
}

func (c *Context) doWith(ctx context.Context, fn func() error, waitAfterSubmit bool) error {
	if c == nil || c.exec == nil {
		return ErrNilContext
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.opMu.RLock()
	defer c.opMu.RUnlock()
	if c.closed.Load() {
		return ErrContextClosed
	}
	if waitAfterSubmit {
		return c.exec.DoCtxWait(ctx, fn)
	}
	return c.exec.DoCtx(ctx, fn)
}

// Close releases the primary context retain and stops the executor. After a
// successful Close, all Context methods return ErrContextClosed and further
// Close calls return nil. If releasing the primary context fails the retain
// count was not dropped, so the Context stays open for Close to be retried.
func (c *Context) Close() error {
	if c == nil || c.exec == nil || c.device == nil {
		return ErrNilContext
	}
	c.opMu.Lock()
	defer c.opMu.Unlock()
	if c.closed.Load() {
		return nil
	}
	var clearErr, releaseErr error
	if err := c.exec.Do(func() error {
		clearErr = cudaresult.CtxSetCurrent(c.driver, 0)
		releaseErr = cudaresult.PrimaryCtxRelease(c.driver, c.device.handle)
		return nil
	}); err != nil {
		return err
	}
	if releaseErr != nil {
		return releaseErr
	}
	c.closed.Store(true)
	_ = c.exec.Close()
	return clearErr
}
