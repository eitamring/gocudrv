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
	device    *Device
	driver    *cudasys.Driver
	raw       cudasys.CUcontext
	exec      *executor.Executor
	opMu      sync.RWMutex
	closed    atomic.Bool
	closeOnce sync.Once
	closeErr  error
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

func (c *Context) closeOnExecutor() error {
	clearErr := cudaresult.CtxSetCurrent(c.driver, 0)
	releaseErr := cudaresult.PrimaryCtxRelease(c.driver, c.device.handle)
	return errors.Join(clearErr, releaseErr)
}

// Close releases the primary context retain and stops the executor.
// Idempotent: subsequent calls return the first call's result. After Close
// returns, all Context methods return ErrContextClosed.
func (c *Context) Close() error {
	if c == nil || c.exec == nil || c.device == nil {
		return ErrNilContext
	}
	c.closeOnce.Do(func() {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		c.closed.Store(true)
		c.closeErr = c.exec.Do(c.closeOnExecutor)
		_ = c.exec.Close()
	})
	return c.closeErr
}
