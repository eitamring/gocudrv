package cuda

import (
	"context"
	"math"
	"sync"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// streamNonBlocking matches CUDA's CU_STREAM_NON_BLOCKING flag from cuda.h.
const streamNonBlocking uint32 = 1

// StreamOption customizes a stream created by Context.NewStream.
type StreamOption interface {
	apply(*streamOptions)
}

type streamOptions struct {
	priority *int32
	err      error
}

type streamOption func(*streamOptions)

func (f streamOption) apply(opts *streamOptions) {
	f(opts)
}

// WithStreamPriority requests a CUDA scheduling priority for the new stream.
// Lower numeric values represent higher priority. CUDA clamps priorities that
// are outside the device's supported range.
func WithStreamPriority(priority int) StreamOption {
	return streamOption(func(opts *streamOptions) {
		if priority < math.MinInt32 || priority > math.MaxInt32 {
			opts.err = ErrInvalidStreamPriority
			return
		}
		p := int32(priority)
		opts.priority = &p
	})
}

// Stream is an ordered queue of CUDA work owned by a Context.
//
// NewStream creates non-blocking streams so work submitted to them does not
// implicitly synchronize with the legacy default stream. A Stream must be
// closed before its owning Context is closed.
type Stream struct {
	ctx    *Context
	raw    cudasys.CUstream
	opMu   sync.RWMutex
	closed bool
}

// NewStream creates a non-blocking CUDA stream owned by the context.
func (c *Context) NewStream(options ...StreamOption) (*Stream, error) {
	if c == nil {
		return nil, ErrNilContext
	}
	opts := streamOptions{}
	for _, option := range options {
		if option == nil {
			continue
		}
		option.apply(&opts)
	}
	if opts.err != nil {
		return nil, opts.err
	}
	var raw cudasys.CUstream
	err := c.doWait(context.Background(), func() error {
		var (
			s cudasys.CUstream
			e error
		)
		if opts.priority != nil {
			s, e = cudaresult.StreamCreateWithPriority(c.driver, streamNonBlocking, *opts.priority)
		} else {
			s, e = cudaresult.StreamCreate(c.driver, streamNonBlocking)
		}
		if e != nil {
			return e
		}
		raw = s
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Stream{ctx: c, raw: raw}, nil
}

// Synchronize waits until all preceding work in the stream has completed or
// ctx is canceled. Canceling ctx stops the caller's wait; queued GPU work and
// the underlying CUDA synchronization continue on the executor thread.
func (s *Stream) Synchronize(ctx context.Context) error {
	if s == nil {
		return ErrNilStream
	}
	s.opMu.RLock()
	defer s.opMu.RUnlock()
	if s.closed {
		return ErrStreamClosed
	}
	return s.ctx.do(ctx, func() error {
		return cudaresult.StreamSynchronize(s.ctx.driver, s.raw)
	})
}

// Close destroys the stream. Idempotent after a successful destroy; failures
// leave the stream open so callers can retry. Returns ErrContextClosed if the
// owning Context was closed first.
//
// Close does not make queued GPU work safe to forget. If a caller closes the
// stream and then frees a buffer or module still used by queued work, the GPU
// may keep touching that freed resource. Synchronize first.
func (s *Stream) Close() error {
	if s == nil {
		return ErrNilStream
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	if s.closed {
		return nil
	}
	if err := s.ctx.doWait(context.Background(), func() error {
		return cudaresult.StreamDestroy(s.ctx.driver, s.raw)
	}); err != nil {
		return err
	}
	s.closed = true
	return nil
}
