package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

type graphCalls struct {
	begin        atomic.Int32
	end          atomic.Int32
	instantiate  atomic.Int32
	launch       atomic.Int32
	graphDestroy atomic.Int32
	execDestroy  atomic.Int32
	lastMode     atomic.Uint32
	lastStream   atomic.Uintptr
}

func graphDriver(c *graphCalls) *cudasys.Driver {
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSetCurrent:           func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamCreate: func(s *cudasys.CUstream, _ uint32) cudasys.CUresult {
			*s = 0x5151
			return cudasys.CUDA_SUCCESS
		},
		CuStreamDestroy: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuStreamBeginCapture: func(_ cudasys.CUstream, mode uint32) cudasys.CUresult {
			c.begin.Add(1)
			c.lastMode.Store(mode)
			return cudasys.CUDA_SUCCESS
		},
		CuStreamEndCapture: func(_ cudasys.CUstream, g *cudasys.CUgraph) cudasys.CUresult {
			c.end.Add(1)
			*g = 0x6A6A
			return cudasys.CUDA_SUCCESS
		},
		CuGraphInstantiate: func(e *cudasys.CUgraphExec, _ cudasys.CUgraph, _ uint64) cudasys.CUresult {
			c.instantiate.Add(1)
			*e = 0x7E7E
			return cudasys.CUDA_SUCCESS
		},
		CuGraphLaunch: func(_ cudasys.CUgraphExec, s cudasys.CUstream) cudasys.CUresult {
			c.launch.Add(1)
			c.lastStream.Store(uintptr(s))
			return cudasys.CUDA_SUCCESS
		},
		CuGraphDestroy: func(cudasys.CUgraph) cudasys.CUresult {
			c.graphDestroy.Add(1)
			return cudasys.CUDA_SUCCESS
		},
		CuGraphExecDestroy: func(cudasys.CUgraphExec) cudasys.CUresult {
			c.execDestroy.Add(1)
			return cudasys.CUDA_SUCCESS
		},
	}
}

func TestGraphCaptureReplayHappy(t *testing.T) {
	var calls graphCalls
	ctx := newTestContext(t, graphDriver(&calls))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	if err := stream.BeginCapture(CaptureModeThreadLocal); err != nil {
		t.Fatalf("BeginCapture: %v", err)
	}
	if calls.lastMode.Load() != 1 {
		t.Errorf("capture mode = %d, want 1 (thread-local)", calls.lastMode.Load())
	}
	g, err := stream.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture: %v", err)
	}
	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate: %v", err)
	}
	if err := exec.Launch(context.Background(), stream); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if calls.begin.Load() != 1 || calls.end.Load() != 1 || calls.instantiate.Load() != 1 || calls.launch.Load() != 1 {
		t.Errorf("calls begin=%d end=%d inst=%d launch=%d, want all 1",
			calls.begin.Load(), calls.end.Load(), calls.instantiate.Load(), calls.launch.Load())
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("launch stream = %#x, want 0x5151", calls.lastStream.Load())
	}

	if err := exec.Close(); err != nil {
		t.Fatalf("exec.Close: %v", err)
	}
	if err := g.Close(); err != nil {
		t.Fatalf("g.Close: %v", err)
	}
	if calls.execDestroy.Load() != 1 || calls.graphDestroy.Load() != 1 {
		t.Errorf("destroy exec=%d graph=%d, want 1 each", calls.execDestroy.Load(), calls.graphDestroy.Load())
	}
	// idempotent close
	if err := exec.Close(); err != nil || calls.execDestroy.Load() != 1 {
		t.Errorf("second exec.Close not idempotent: err=%v destroys=%d", err, calls.execDestroy.Load())
	}
}

func TestGraphRejects(t *testing.T) {
	var calls graphCalls
	ctx := newTestContext(t, graphDriver(&calls))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	closedStream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream closed: %v", err)
	}
	if err := closedStream.Close(); err != nil {
		t.Fatalf("close closedStream: %v", err)
	}

	// build a real exec to test its reject paths
	if err := stream.BeginCapture(CaptureModeGlobal); err != nil {
		t.Fatalf("BeginCapture: %v", err)
	}
	g, err := stream.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture: %v", err)
	}
	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate: %v", err)
	}

	other := newTestContext(t, graphDriver(&graphCalls{}))
	otherStream, err := other.NewStream()
	if err != nil {
		t.Fatalf("other NewStream: %v", err)
	}
	t.Cleanup(func() { _ = otherStream.Close() })

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"begincapture nil stream", func() error { var s *Stream; return s.BeginCapture(CaptureModeGlobal) }, ErrNilStream},
		{"begincapture closed stream", func() error { return closedStream.BeginCapture(CaptureModeGlobal) }, ErrStreamClosed},
		{"endcapture nil stream", func() error { var s *Stream; _, e := s.EndCapture(); return e }, ErrNilStream},
		{"instantiate nil graph", func() error { var gg *Graph; _, e := gg.Instantiate(); return e }, ErrNilGraph},
		{"launch nil exec", func() error { var e *GraphExec; return e.Launch(context.Background(), stream) }, ErrNilGraphExec},
		{"launch nil stream", func() error { return exec.Launch(context.Background(), nil) }, ErrNilStream},
		{"launch closed stream", func() error { return exec.Launch(context.Background(), closedStream) }, ErrStreamClosed},
		{"launch wrong context stream", func() error { return exec.Launch(context.Background(), otherStream) }, ErrContextMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestGraphClosedRejects(t *testing.T) {
	var calls graphCalls
	ctx := newTestContext(t, graphDriver(&calls))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	if err := stream.BeginCapture(CaptureModeGlobal); err != nil {
		t.Fatalf("BeginCapture: %v", err)
	}
	g, err := stream.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture: %v", err)
	}
	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate: %v", err)
	}
	if err := exec.Close(); err != nil {
		t.Fatalf("exec.Close: %v", err)
	}
	if err := exec.Launch(context.Background(), stream); !errors.Is(err, ErrGraphExecClosed) {
		t.Errorf("launch closed exec = %v, want ErrGraphExecClosed", err)
	}
	if err := g.Close(); err != nil {
		t.Fatalf("g.Close: %v", err)
	}
	if _, err := g.Instantiate(); !errors.Is(err, ErrGraphClosed) {
		t.Errorf("instantiate closed graph = %v, want ErrGraphClosed", err)
	}
}

func TestGraphPropagatesError(t *testing.T) {
	var calls graphCalls
	ctx := newTestContext(t, graphDriver(&calls))
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	ctx.driver.CuStreamBeginCapture = func(cudasys.CUstream, uint32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if err := stream.BeginCapture(CaptureModeGlobal); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("BeginCapture err = %v, want ErrInvalidValue", err)
	}
}
