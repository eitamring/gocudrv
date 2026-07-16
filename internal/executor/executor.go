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
	result *completion
}

type completionState uint32

const (
	completionWaiting completionState = iota
	completionAbandoned
	completionReceived
	completionDelivered
)

// completion coordinates ownership of a result channel after either side can
// finish first. The side that observes the other's state returns it to the pool.
type completion struct {
	result chan error
	state  atomic.Uint32
}

func (c *completion) delivered() bool {
	previous := completionState(c.state.Swap(uint32(completionDelivered)))
	if previous == completionAbandoned {
		<-c.result
	}
	return previous != completionWaiting
}

func (c *completion) received() bool {
	return completionState(c.state.Swap(uint32(completionReceived))) == completionDelivered
}

func (c *completion) abandon() bool {
	if completionState(c.state.Swap(uint32(completionAbandoned))) != completionDelivered {
		return false
	}
	<-c.result
	return true
}

// resultPool recycles completions so a submission on a hot path does not
// allocate. The ownership handshake guarantees every pooled channel is empty,
// including when a caller cancels after submission.
var resultPool = sync.Pool{New: func() any {
	return &completion{result: make(chan error, 1)}
}}

func recycleCompletion(c *completion) {
	c.state.Store(uint32(completionWaiting))
	resultPool.Put(c)
}

// Executor runs functions on a single OS thread. Construct one per CUDA
// context so that "current context" stays stable across calls. Close normally
// unlocks the thread for reuse. Retire keeps an unclean thread locked and
// parked for the rest of the process so it is neither reused nor terminated.
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
	workerSpin           = 4096
	callerSpin           = 4096
	multiPSpinYieldEvery = 128
)

func spinYieldEvery() int {
	return spinYieldEveryFor(runtime.GOMAXPROCS(0))
}

func spinYieldEveryFor(procs int) int {
	if procs <= 1 {
		return 1
	}
	return multiPSpinYieldEvery
}

func shouldYield(spins, every int) bool {
	return spins > 0 && spins%every == 0
}

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
	defer e.finishThread()
	spin := 0
	yieldEvery := multiPSpinYieldEvery
	for {
		select {
		case t := <-e.tasks:
			e.runTask(t)
			spin = workerSpin
			yieldEvery = spinYieldEvery()
			continue
		case <-e.quit:
			e.drain()
			return
		default:
		}
		if spin > 0 {
			spin--
			if shouldYield(spin, yieldEvery) {
				runtime.Gosched()
			}
			continue
		}
		select {
		case t := <-e.tasks:
			e.runTask(t)
			spin = workerSpin
			yieldEvery = spinYieldEvery()
		case <-e.quit:
			e.drain()
			return
		}
	}
}

func (e *Executor) finishThread() {
	if e.retire.Load() {
		// Let Close return before parking the locked thread for good.
		close(e.done)
		quarantineThread()
	}
	runtime.UnlockOSThread()
	close(e.done)
}

// quarantineThread deliberately owns its locked OS thread until process exit.
func quarantineThread() { select {} }

func (e *Executor) runTask(t task) {
	err := e.runOne(t.job)
	RecycleJob(t.job)
	t.result.result <- err
	if t.result.delivered() {
		recycleCompletion(t.result)
	}
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

// Retire marks the executor's thread as unclean. Close then quarantines the
// locked thread for the rest of the process instead of terminating or reusing
// it. Call it only when foreign thread-local state could not be cleared.
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

func (e *Executor) submit(ctx context.Context, j Job) (*completion, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	res := resultPool.Get().(*completion)
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		recycleCompletion(res)
		return nil, ErrExecutorClosed
	}
	select {
	case e.tasks <- task{job: j, result: res}:
	case <-ctx.Done():
		e.mu.RUnlock()
		recycleCompletion(res)
		return nil, ctx.Err()
	case <-e.done:
		e.mu.RUnlock()
		recycleCompletion(res)
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
	if ctx == nil {
		ctx = context.Background()
	}
	res, err := e.submit(ctx, j)
	if err != nil {
		RecycleJob(j)
		return err
	}
	yieldEvery := spinYieldEvery()
	for i := 0; i < callerSpin; i++ {
		select {
		case err := <-res.result:
			if res.received() {
				recycleCompletion(res)
			}
			return err
		case <-ctx.Done():
			if res.abandon() {
				recycleCompletion(res)
			}
			return ctx.Err()
		default:
		}
		if shouldYield(i+1, yieldEvery) {
			runtime.Gosched()
		}
	}
	select {
	case err := <-res.result:
		if res.received() {
			recycleCompletion(res)
		}
		return err
	case <-ctx.Done():
		if res.abandon() {
			recycleCompletion(res)
		}
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
	yieldEvery := spinYieldEvery()
	for i := 0; i < callerSpin; i++ {
		select {
		case err := <-res.result:
			if res.received() {
				recycleCompletion(res)
			}
			return err
		default:
		}
		if shouldYield(i+1, yieldEvery) {
			runtime.Gosched()
		}
	}
	err = <-res.result
	if res.received() {
		recycleCompletion(res)
	}
	return err
}

// Close stops accepting work and waits for every accepted task to finish. A
// clean worker exits; a retired worker signals completion and then stays
// quarantined. Idempotent; the first call's error (if any) is returned on every
// subsequent call.
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
