package executor

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
)

// ErrExecutorClosed is returned by Do/DoCtx when the executor has been
// closed or is in the process of closing.
var ErrExecutorClosed = errors.New("cuda: executor is closed")

// PanicError wraps a value recovered from a function that panicked inside
// the executor goroutine. The executor stays alive after a panic so the
// caller can decide whether to keep using it or close it.
type PanicError struct {
	Value any
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("cuda: executor panic: %v", e.Value)
}

// Is matches any *PanicError regardless of the recovered value, so callers
// can write errors.Is(err, &executor.PanicError{}).
func (e *PanicError) Is(target error) bool {
	_, ok := target.(*PanicError)
	return ok
}

type task struct {
	fn     func() error
	result chan error
}

// resultPool recycles the per-call completion channels so a submission on a hot
// path does not allocate one each time. A channel is only returned to the pool
// once its result has been received (or it was never handed to the executor), so
// every channel taken from the pool is empty.
var resultPool = sync.Pool{New: func() any { return make(chan error, 1) }}

// Executor runs functions on a single OS thread. Construct one per CUDA
// context so that "current context" stays stable across calls. The pinned
// goroutine never unlocks its OS thread; when Close stops it, the Go
// runtime retires the thread automatically.
type Executor struct {
	tasks     chan task
	quit      chan struct{}
	done      chan struct{}
	mu        sync.RWMutex
	closed    bool
	closeOnce sync.Once
	closeErr  error
}

// New starts a pinned-thread executor goroutine.
func New() *Executor {
	e := &Executor{
		tasks: make(chan task),
		quit:  make(chan struct{}),
		done:  make(chan struct{}),
	}
	go e.run()
	return e
}

func (e *Executor) run() {
	runtime.LockOSThread()
	defer close(e.done)
	for {
		select {
		case t := <-e.tasks:
			t.result <- e.runOne(t.fn)
		case <-e.quit:
			return
		}
	}
}

func (e *Executor) runOne(fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = &PanicError{Value: r}
		}
	}()
	return fn()
}

// Do is shorthand for DoCtx(context.Background(), fn). Use it when there is
// no meaningful cancellation point.
func (e *Executor) Do(fn func() error) error {
	return e.DoCtx(context.Background(), fn)
}

func (e *Executor) submit(ctx context.Context, fn func() error) (chan error, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	res := resultPool.Get().(chan error)
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		resultPool.Put(res)
		return nil, ErrExecutorClosed
	}
	select {
	case e.tasks <- task{fn: fn, result: res}:
	case <-ctx.Done():
		e.mu.RUnlock()
		resultPool.Put(res)
		return nil, ctx.Err()
	case <-e.done:
		e.mu.RUnlock()
		resultPool.Put(res)
		return nil, ErrExecutorClosed
	}
	e.mu.RUnlock()
	return res, nil
}

// DoCtx runs fn on the executor's pinned thread and blocks until either fn
// returns or ctx is canceled. If ctx is canceled, DoCtx returns ctx.Err()
// even though fn may still be running on the executor; the result is then
// discarded. Returns ErrExecutorClosed if the executor is closed before or
// during the call. Panics inside fn are recovered and surfaced as
// *PanicError; the executor keeps running.
func (e *Executor) DoCtx(ctx context.Context, fn func() error) error {
	res, err := e.submit(ctx, fn)
	if err != nil {
		return err
	}
	select {
	case err := <-res:
		resultPool.Put(res)
		return err
	case <-ctx.Done():
		// fn is still in flight and the executor will send to res later, so the
		// channel cannot be recycled; let it be collected instead.
		return ctx.Err()
	}
}

// DoCtxWait runs fn on the executor's pinned thread. ctx may prevent
// submission, but once fn is submitted DoCtxWait waits for it to finish. Use it
// for operations that pass Go memory to foreign code, where returning early
// would let the caller reuse memory while work is still in flight.
func (e *Executor) DoCtxWait(ctx context.Context, fn func() error) error {
	res, err := e.submit(ctx, fn)
	if err != nil {
		return err
	}
	err = <-res
	resultPool.Put(res)
	return err
}

// Close stops the executor goroutine and waits for it to exit, including
// any task that is currently running. Idempotent; the first call's error
// (if any) is returned on every subsequent call.
func (e *Executor) Close() error {
	e.closeOnce.Do(func() {
		e.mu.Lock()
		e.closed = true
		close(e.quit)
		e.mu.Unlock()
		<-e.done
	})
	return e.closeErr
}
