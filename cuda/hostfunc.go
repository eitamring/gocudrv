package cuda

import (
	"context"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/internal/hostcb"
)

// LaunchHostFunc enqueues fn on the stream: the driver calls it on one of its
// internal threads once all preceding stream work completes, and later stream
// work waits for it to return. fn must not call back into CUDA (or block on
// work from this stream) and should be short; a panic inside fn is swallowed.
// Returns ErrSymbolUnavailable on a driver without cuLaunchHostFunc.
func (s *Stream) LaunchHostFunc(fn func()) error {
	if s == nil {
		return ErrNilStream
	}
	if fn == nil {
		return ErrNilHostFunc
	}
	s.opMu.RLock()
	defer s.opMu.RUnlock()
	if s.closed {
		return ErrStreamClosed
	}
	key := hostcb.Register(fn)
	err := s.ctx.doWait(context.Background(), func() error {
		return cudaresult.LaunchHostFunc(s.ctx.driver, s.raw, hostcb.Callback(), key)
	})
	if err != nil {
		hostcb.Unregister(key)
	}
	return err
}
