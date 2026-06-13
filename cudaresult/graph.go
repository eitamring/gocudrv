package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// StreamBeginCapture puts stream into capture mode. Work enqueued on the stream
// after this call is recorded into a graph instead of being executed, until
// StreamEndCapture is called. mode is a CUstreamCaptureMode value.
func StreamBeginCapture(d *cudasys.Driver, stream cudasys.CUstream, mode uint32) error {
	if d == nil || d.CuStreamBeginCapture == nil {
		return ErrNotInitialized
	}
	return check("cuStreamBeginCapture_v2", d.CuStreamBeginCapture(stream, mode))
}

// StreamEndCapture ends capture on stream and returns the recorded graph.
func StreamEndCapture(d *cudasys.Driver, stream cudasys.CUstream) (cudasys.CUgraph, error) {
	if d == nil || d.CuStreamEndCapture == nil {
		return 0, ErrNotInitialized
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
	if d == nil || d.CuGraphInstantiate == nil {
		return 0, ErrNotInitialized
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
	if d == nil || d.CuGraphLaunch == nil {
		return ErrNotInitialized
	}
	return check("cuGraphLaunch", d.CuGraphLaunch(exec, stream))
}

// GraphDestroy releases a graph previously returned by StreamEndCapture.
func GraphDestroy(d *cudasys.Driver, graph cudasys.CUgraph) error {
	if d == nil || d.CuGraphDestroy == nil {
		return ErrNotInitialized
	}
	return check("cuGraphDestroy", d.CuGraphDestroy(graph))
}

// GraphExecDestroy releases an executable graph previously returned by
// GraphInstantiate.
func GraphExecDestroy(d *cudasys.Driver, exec cudasys.CUgraphExec) error {
	if d == nil || d.CuGraphExecDestroy == nil {
		return ErrNotInitialized
	}
	return check("cuGraphExecDestroy", d.CuGraphExecDestroy(exec))
}
