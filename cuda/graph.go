package cuda

import (
	"context"
	"sync"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// CaptureMode selects how stream capture interacts with other threads. It
// mirrors CUstreamCaptureMode.
type CaptureMode uint32

const (
	// CaptureModeGlobal is the default: capture is visible across the process
	// and unsafe operations elsewhere invalidate the capture.
	CaptureModeGlobal CaptureMode = 0
	// CaptureModeThreadLocal restricts the safety checks to the calling thread.
	CaptureModeThreadLocal CaptureMode = 1
	// CaptureModeRelaxed disables the safety checks.
	CaptureModeRelaxed CaptureMode = 2
)

// Graph is a recorded sequence of GPU work captured from a stream. Instantiate
// it into a GraphExec to launch it. Its lifetime is bound to the Context.
//
// Lifetime rule: a Graph must be closed before its owning Context is closed.
type Graph struct {
	ctx    *Context
	raw    cudasys.CUgraph
	opMu   sync.RWMutex
	closed bool
}

// GraphExec is an executable graph compiled from a Graph. Launch it repeatedly
// on a stream with near-zero per-launch overhead. Its lifetime is bound to the
// Context.
type GraphExec struct {
	ctx    *Context
	raw    cudasys.CUgraphExec
	opMu   sync.RWMutex
	closed bool
}

// BeginCapture puts the stream into capture mode. Work enqueued on the stream
// after this call is recorded into a graph instead of running, until
// EndCapture is called. Use CaptureModeGlobal unless you have a reason not to.
func (s *Stream) BeginCapture(mode CaptureMode) error {
	if s == nil {
		return ErrNilStream
	}
	s.opMu.RLock()
	defer s.opMu.RUnlock()
	if s.closed {
		return ErrStreamClosed
	}
	return s.ctx.doWait(context.Background(), func() error {
		return cudaresult.StreamBeginCapture(s.ctx.driver, s.raw, uint32(mode))
	})
}

// EndCapture ends capture on the stream and returns the recorded Graph. The
// caller owns the Graph and must Close it before the Context.
func (s *Stream) EndCapture() (*Graph, error) {
	if s == nil {
		return nil, ErrNilStream
	}
	s.opMu.RLock()
	defer s.opMu.RUnlock()
	if s.closed {
		return nil, ErrStreamClosed
	}
	var raw cudasys.CUgraph
	err := s.ctx.doWait(context.Background(), func() error {
		g, e := cudaresult.StreamEndCapture(s.ctx.driver, s.raw)
		if e != nil {
			return e
		}
		raw = g
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Graph{ctx: s.ctx, raw: raw}, nil
}

// Instantiate compiles the graph into an executable GraphExec. The caller owns
// the GraphExec and must Close it before the Context.
func (g *Graph) Instantiate() (*GraphExec, error) {
	if g == nil {
		return nil, ErrNilGraph
	}
	g.opMu.Lock()
	defer g.opMu.Unlock()
	if g.closed {
		return nil, ErrGraphClosed
	}
	var raw cudasys.CUgraphExec
	err := g.ctx.doWait(context.Background(), func() error {
		e, ee := cudaresult.GraphInstantiate(g.ctx.driver, g.raw, 0)
		if ee != nil {
			return ee
		}
		raw = e
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &GraphExec{ctx: g.ctx, raw: raw}, nil
}

// Close releases the graph. Idempotent after a successful destroy. Returns
// ErrContextClosed if the owning Context was closed first.
func (g *Graph) Close() error {
	if g == nil {
		return ErrNilGraph
	}
	g.opMu.Lock()
	defer g.opMu.Unlock()
	if g.closed {
		return nil
	}
	if err := g.ctx.doWait(context.Background(), func() error {
		return cudaresult.GraphDestroy(g.ctx.driver, g.raw)
	}); err != nil {
		return err
	}
	g.closed = true
	return nil
}

// Launch enqueues the executable graph on stream. It returns after the driver
// accepts the work, not after the GPU finishes; call stream.Synchronize before
// reading outputs. The stream must belong to the same Context as the graph.
func (e *GraphExec) Launch(ctx context.Context, stream *Stream) error {
	if e == nil {
		return ErrNilGraphExec
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	if e.closed {
		return ErrGraphExecClosed
	}
	if stream.ctx != e.ctx {
		return ErrContextMismatch
	}
	return e.ctx.doWait(ctx, func() error {
		return cudaresult.GraphLaunch(e.ctx.driver, e.raw, stream.raw)
	})
}

// Update re-applies graph's node parameters to this executable without
// re-instantiating. It returns ErrGraphExecUpdateFailure if the topology changed
// too much to update; re-instantiate then. graph must share the Context.
func (e *GraphExec) Update(graph *Graph) error {
	if e == nil {
		return ErrNilGraphExec
	}
	if graph == nil {
		return ErrNilGraph
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	if e.closed {
		return ErrGraphExecClosed
	}
	graph.opMu.Lock()
	defer graph.opMu.Unlock()
	if graph.closed {
		return ErrGraphClosed
	}
	if graph.ctx != e.ctx {
		return ErrContextMismatch
	}
	return e.ctx.doWait(context.Background(), func() error {
		return cudaresult.GraphExecUpdate(e.ctx.driver, e.raw, graph.raw)
	})
}

// Close releases the executable graph. Idempotent after a successful destroy.
// Returns ErrContextClosed if the owning Context was closed first.
func (e *GraphExec) Close() error {
	if e == nil {
		return ErrNilGraphExec
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	if e.closed {
		return nil
	}
	if err := e.ctx.doWait(context.Background(), func() error {
		return cudaresult.GraphExecDestroy(e.ctx.driver, e.raw)
	}); err != nil {
		return err
	}
	e.closed = true
	return nil
}
