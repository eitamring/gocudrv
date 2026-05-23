package cuda

import (
	"context"
	"sync"
	"time"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

const (
	eventDefault       uint32 = 0
	eventBlockingSync  uint32 = 1
	eventDisableTiming uint32 = 2
)

// waitEventDefault matches CUDA's CU_EVENT_WAIT_DEFAULT flag.
const waitEventDefault uint32 = 0

// EventOption customizes an event created by Context.NewEvent.
type EventOption interface {
	apply(*eventOptions)
}

type eventOptions struct {
	flags uint32
	err   error
}

type eventOptionFunc func(*eventOptions)

func (f eventOptionFunc) apply(opts *eventOptions) {
	f(opts)
}

// WithEventBlockingSync requests blocking host synchronization for the event
// instead of CUDA's default spin-wait behavior.
func WithEventBlockingSync() EventOption {
	return eventOptionFunc(func(opts *eventOptions) {
		if opts.err != nil {
			return
		}
		opts.flags |= eventBlockingSync
	})
}

// WithEventDisableTiming disables timestamp recording for the event. Use this
// for events that only order stream work and will not be passed to Elapsed.
func WithEventDisableTiming() EventOption {
	return eventOptionFunc(func(opts *eventOptions) {
		if opts.err != nil {
			return
		}
		opts.flags |= eventDisableTiming
	})
}

// Event is a CUDA event owned by a Context. Events mark positions in streams,
// allow other streams to wait on those positions, and can measure elapsed GPU
// time between two recorded events.
type Event struct {
	ctx            *Context
	raw            cudasys.CUevent
	opMu           sync.RWMutex
	closed         bool
	timingDisabled bool
}

// NewEvent creates a CUDA event owned by the context. Close the returned event
// before closing the context.
func (c *Context) NewEvent(options ...EventOption) (*Event, error) {
	if c == nil {
		return nil, ErrNilContext
	}
	opts := eventOptions{flags: eventDefault}
	for _, option := range options {
		if option == nil {
			continue
		}
		option.apply(&opts)
		if opts.err != nil {
			return nil, opts.err
		}
	}
	var raw cudasys.CUevent
	err := c.doWait(context.Background(), func() error {
		e, err := cudaresult.EventCreate(c.driver, opts.flags)
		if err != nil {
			return err
		}
		raw = e
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Event{ctx: c, raw: raw, timingDisabled: opts.flags&eventDisableTiming != 0}, nil
}

// Record enqueues this event into stream. Work later submitted to the same
// stream runs after the event; other streams can wait on it with
// Stream.WaitEvent.
func (e *Event) Record(stream *Stream) error {
	if e == nil {
		return ErrNilEvent
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	e.opMu.RLock()
	defer e.opMu.RUnlock()
	if e.closed {
		return ErrEventClosed
	}
	if stream.ctx != e.ctx {
		return ErrContextMismatch
	}
	return e.ctx.doWait(context.Background(), func() error {
		return cudaresult.EventRecord(e.ctx.driver, e.raw, stream.raw)
	})
}

// Synchronize waits until the event has completed or ctx is canceled.
// Canceling ctx stops the caller's wait; the underlying CUDA synchronization
// continues on the context executor.
func (e *Event) Synchronize(ctx context.Context) error {
	if e == nil {
		return ErrNilEvent
	}
	e.opMu.RLock()
	defer e.opMu.RUnlock()
	if e.closed {
		return ErrEventClosed
	}
	return e.ctx.do(ctx, func() error {
		return cudaresult.EventSynchronize(e.ctx.driver, e.raw)
	})
}

// Query reports whether the event has completed. It returns nil when the event
// is ready, ErrNotReady when CUDA reports it is still pending, and another
// CUDA error for driver failures.
func (e *Event) Query() error {
	if e == nil {
		return ErrNilEvent
	}
	e.opMu.RLock()
	defer e.opMu.RUnlock()
	if e.closed {
		return ErrEventClosed
	}
	return e.ctx.doWait(context.Background(), func() error {
		return cudaresult.EventQuery(e.ctx.driver, e.raw)
	})
}

// Elapsed returns the GPU time between this event and end. Both events must be
// recorded and completed before calling Elapsed; otherwise CUDA may return
// ErrNotReady.
func (e *Event) Elapsed(end *Event) (time.Duration, error) {
	if e == nil || end == nil {
		return 0, ErrNilEvent
	}
	unlock := lockEvents(e, end)
	defer unlock()
	if e.closed || end.closed {
		return 0, ErrEventClosed
	}
	if e.timingDisabled || end.timingDisabled {
		return 0, ErrEventTimingDisabled
	}
	if e.ctx != end.ctx {
		return 0, ErrContextMismatch
	}
	var ms float32
	err := e.ctx.doWait(context.Background(), func() error {
		v, err := cudaresult.EventElapsedTime(e.ctx.driver, e.raw, end.raw)
		if err != nil {
			return err
		}
		ms = v
		return nil
	})
	if err != nil {
		return 0, err
	}
	return time.Duration(float64(ms) * float64(time.Millisecond)), nil
}

// Close destroys the event. Idempotent after a successful destroy; failures
// leave the event open so callers can retry. Returns ErrContextClosed if the
// owning Context was closed first.
func (e *Event) Close() error {
	if e == nil {
		return ErrNilEvent
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	if e.closed {
		return nil
	}
	if err := e.ctx.doWait(context.Background(), func() error {
		return cudaresult.EventDestroy(e.ctx.driver, e.raw)
	}); err != nil {
		return err
	}
	e.closed = true
	return nil
}

func lockEvents(a, b *Event) func() {
	if a == b {
		a.opMu.RLock()
		return a.opMu.RUnlock
	}
	// Lock by address so concurrent a.Elapsed(b) and b.Elapsed(a) cannot deadlock.
	first, second := a, b
	if uintptr(unsafe.Pointer(first)) > uintptr(unsafe.Pointer(second)) {
		first, second = second, first
	}
	first.opMu.RLock()
	second.opMu.RLock()
	return func() {
		second.opMu.RUnlock()
		first.opMu.RUnlock()
	}
}
