package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// StreamBeginCapture puts stream into capture mode. Work enqueued on the stream
// after this call is recorded into a graph instead of being executed, until
// StreamEndCapture is called. mode is a CUstreamCaptureMode value.
func StreamBeginCapture(d *cudasys.Driver, stream cudasys.CUstream, mode uint32) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuStreamBeginCapture == nil {
		return ErrSymbolUnavailable
	}
	return check("cuStreamBeginCapture_v2", d.CuStreamBeginCapture(stream, mode))
}

// StreamEndCapture ends capture on stream and returns the recorded graph.
func StreamEndCapture(d *cudasys.Driver, stream cudasys.CUstream) (cudasys.CUgraph, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuStreamEndCapture == nil {
		return 0, ErrSymbolUnavailable
	}
	var graph cudasys.CUgraph
	if err := check("cuStreamEndCapture", d.CuStreamEndCapture(stream, &graph)); err != nil {
		return 0, err
	}
	return graph, nil
}

// GraphInstantiate compiles a captured graph into an executable graph that can
// be launched repeatedly. flags are CUgraphInstantiate flags; pass 0 for none.
func GraphInstantiate(d *cudasys.Driver, graph cudasys.CUgraph, flags uint64) (cudasys.CUgraphExec, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuGraphInstantiate == nil {
		return 0, ErrSymbolUnavailable
	}
	var exec cudasys.CUgraphExec
	if err := check("cuGraphInstantiateWithFlags", d.CuGraphInstantiate(&exec, graph, flags)); err != nil {
		return 0, err
	}
	return exec, nil
}

// GraphLaunch enqueues an executable graph on stream. It returns after the
// driver accepts the work, not after the GPU finishes.
func GraphLaunch(d *cudasys.Driver, exec cudasys.CUgraphExec, stream cudasys.CUstream) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuGraphLaunch == nil {
		return ErrSymbolUnavailable
	}
	return check("cuGraphLaunch", d.CuGraphLaunch(exec, stream))
}

// GraphDestroy releases a graph previously returned by StreamEndCapture.
func GraphDestroy(d *cudasys.Driver, graph cudasys.CUgraph) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuGraphDestroy == nil {
		return ErrSymbolUnavailable
	}
	return check("cuGraphDestroy", d.CuGraphDestroy(graph))
}

// graphExecUpdateSuccess is CU_GRAPH_EXEC_UPDATE_SUCCESS.
const graphExecUpdateSuccess int32 = 0

// GraphExecUpdate re-applies graph's node parameters to exec without
// re-instantiating, using the legacy four-argument cuGraphExecUpdate. It returns
// ErrGraphExecUpdateFailure when the driver declines the update, which it can
// signal through the result out-parameter even on a CUDA_SUCCESS status, and
// ErrSymbolUnavailable on a driver that lacks the best-effort symbol.
func GraphExecUpdate(d *cudasys.Driver, exec cudasys.CUgraphExec, graph cudasys.CUgraph) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuGraphExecUpdate == nil {
		return ErrSymbolUnavailable
	}
	var errNode cudasys.CUgraphNode
	var result int32
	if err := check("cuGraphExecUpdate", d.CuGraphExecUpdate(exec, graph, &errNode, &result)); err != nil {
		return err
	}
	if result != graphExecUpdateSuccess {
		return ErrGraphExecUpdateFailure
	}
	return nil
}

// GraphExecDestroy releases an executable graph previously returned by
// GraphInstantiate.
func GraphExecDestroy(d *cudasys.Driver, exec cudasys.CUgraphExec) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuGraphExecDestroy == nil {
		return ErrSymbolUnavailable
	}
	return check("cuGraphExecDestroy", d.CuGraphExecDestroy(exec))
}
