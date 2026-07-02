package executor

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
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

// Job is a unit of work run on the executor's pinned thread. Implementing it
// with a pooled pointer receiver lets a hot path submit work without allocating
// a closure per call; DoCtx/DoCtxWait wrap a plain func in funcJob for the
// callers that do not need that. A Job that also implements Recycle() is
// returned to its pool by the worker right after Run, which lets a cancellable
// caller abandon the result without racing the pool.
type Job interface {
	Run() error
}

// funcJob adapts a func to Job. Converting a func value to funcJob and passing
// it as a Job interface does not allocate, so the closure-based methods stay as
// cheap as before.
type funcJob func() error

func (f funcJob) Run() error { return f() }

type task struct {
	job    Job
	result chan error
}

// resultPool recycles the per-call completion channels so a submission on a hot
// path does not allocate one each time. A channel is only returned to the pool
// once its result has been received (or it was never handed to the executor), so
// every channel taken from the pool is empty.
var resultPool = sync.Pool{New: func() any { return make(chan error, 1) }}

// Executor runs functions on a single OS thread. Construct one per CUDA
// context so that "current context" stays stable across calls. When Close stops
// the goroutine it unlocks the thread back to the scheduler instead of retiring
// it: terminating a thread that holds CUDA driver TLS can crash the driver.
// Retire opts back into termination for a thread whose state could not be
// cleared.
type Executor struct {
	tasks     chan task
	quit      chan struct{}
	done      chan struct{}
	mu        sync.RWMutex
	closed    bool
	closeOnce sync.Once
	closeErr  error
	retire    atomic.Bool
}

// Both sides of the handoff spin for these bounded windows (poll iterations)
// before parking, because waking a parked goroutine across the pinned thread
// costs far more than the spin (see docs/internals.md). Idle executors park.
const (
	workerSpin = 4096
	callerSpin = 4096
)

// New starts a pinned-thread executor goroutine.
func New() *Executor {
	e := &Executor{
		tasks: make(chan task, 1),
		quit:  make(chan struct{}),
		done:  make(chan struct{}),
	}
	go e.run()
	return e
}

func (e *Executor) run() {
	runtime.LockOSThread()
	defer close(e.done)
	defer func() {
		if !e.retire.Load() {
			runtime.UnlockOSThread()
		}
	}()
	spin := 0
	for {
		select {
		case t := <-e.tasks:
			e.runTask(t)
			spin = workerSpin
			continue
		case <-e.quit:
			e.drain()
			return
		default:
		}
		if spin > 0 {
			spin--
			continue
		}
		select {
		case t := <-e.tasks:
			e.runTask(t)
			spin = workerSpin
		case <-e.quit:
			e.drain()
			return
		}
	}
}

func (e *Executor) runTask(t task) {
	err := e.runOne(t.job)
	RecycleJob(t.job)
	t.result <- err
}

// RecycleJob returns a worker-owned job to its pool if it implements Recycle.
// Call it wherever a job is rejected before the worker could accept it.
func RecycleJob(j Job) {
	if r, ok := j.(interface{ Recycle() }); ok {
		r.Recycle()
	}
}

// drain runs tasks already accepted into the buffered channel when quit fires.
// submit completes its send before Close can flip closed, so a task can sit in
// the buffer while quit is also ready; abandoning it would hang its caller.
func (e *Executor) drain() {
	for {
		select {
		case t := <-e.tasks:
			e.runTask(t)
		default:
			return
		}
	}
}

// Retire marks the executor's thread as unclean, so Close terminates it instead
// of unlocking it back to the scheduler. Call it when foreign thread-local
// state could not be cleared and must not leak to other goroutines.
func (e *Executor) Retire() {
	e.retire.Store(true)
}

func (e *Executor) runOne(j Job) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = &PanicError{Value: r}
		}
	}()
	return j.Run()
}

// Do is shorthand for DoCtx(context.Background(), fn). Use it when there is
// no meaningful cancellation point.
func (e *Executor) Do(fn func() error) error {
	return e.DoCtx(context.Background(), fn)
}

func (e *Executor) submit(ctx context.Context, j Job) (chan error, error) {
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
	case e.tasks <- task{job: j, result: res}:
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
	return e.DoJobCtx(ctx, funcJob(fn))
}

// DoJobCtx runs j with the cancellation semantics of DoCtx: ctx can abandon
// the wait while j still runs on the executor. A job used this way must be
// recycled by the executor (implement Recycle), never by the caller.
func (e *Executor) DoJobCtx(ctx context.Context, j Job) error {
	res, err := e.submit(ctx, j)
	if err != nil {
		RecycleJob(j)
		return err
	}
	for i := 0; i < callerSpin; i++ {
		select {
		case err := <-res:
			resultPool.Put(res)
			return err
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}
	select {
	case err := <-res:
		resultPool.Put(res)
		return err
	case <-ctx.Done():
		// The job is still in flight and the executor will send to res later, so
		// the channel cannot be recycled; let it be collected instead.
		return ctx.Err()
	}
}

// DoCtxWait runs fn on the executor's pinned thread. ctx may prevent
// submission, but once fn is submitted DoCtxWait waits for it to finish. Use it
// for operations that pass Go memory to foreign code, where returning early
// would let the caller reuse memory while work is still in flight.
func (e *Executor) DoCtxWait(ctx context.Context, fn func() error) error {
	return e.DoJob(ctx, funcJob(fn))
}

// DoJob runs j on the executor's pinned thread and, like DoCtxWait, waits for it
// to finish once it is submitted. ctx may still prevent submission. It lets a
// caller submit a pooled Job instead of a closure, so a hot path allocates
// nothing per call. Panics inside Run are recovered and surfaced as *PanicError.
func (e *Executor) DoJob(ctx context.Context, j Job) error {
	res, err := e.submit(ctx, j)
	if err != nil {
		RecycleJob(j)
		return err
	}
	for i := 0; i < callerSpin; i++ {
		select {
		case err := <-res:
			resultPool.Put(res)
			return err
		default:
		}
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
