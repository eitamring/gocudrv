package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestStreamBeginCapture(t *testing.T) {
	if err := StreamBeginCapture(nil, 0x5151, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := StreamBeginCapture(&cudasys.Driver{}, 0x5151, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var gotMode uint32
	d := &cudasys.Driver{CuStreamBeginCapture: func(_ cudasys.CUstream, mode uint32) cudasys.CUresult {
		gotMode = mode
		return cudasys.CUDA_SUCCESS
	}}
	if err := StreamBeginCapture(d, 0x5151, 2); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if gotMode != 2 {
		t.Errorf("mode = %d, want 2", gotMode)
	}
	dErr := &cudasys.Driver{CuStreamBeginCapture: func(cudasys.CUstream, uint32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}}
	if err := StreamBeginCapture(dErr, 0x5151, 0); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}

func TestStreamEndCapture(t *testing.T) {
	if _, err := StreamEndCapture(nil, 0x5151); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := StreamEndCapture(&cudasys.Driver{}, 0x5151); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	d := &cudasys.Driver{CuStreamEndCapture: func(_ cudasys.CUstream, g *cudasys.CUgraph) cudasys.CUresult {
		*g = 0x6A6A
		return cudasys.CUDA_SUCCESS
	}}
	g, err := StreamEndCapture(d, 0x5151)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if g != 0x6A6A {
		t.Errorf("graph = %#x, want 0x6A6A", g)
	}
}

func TestGraphInstantiate(t *testing.T) {
	if _, err := GraphInstantiate(nil, 0x6A6A, 0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := GraphInstantiate(&cudasys.Driver{}, 0x6A6A, 0); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	d := &cudasys.Driver{CuGraphInstantiate: func(e *cudasys.CUgraphExec, _ cudasys.CUgraph, _ uint64) cudasys.CUresult {
		*e = 0x7E7E
		return cudasys.CUDA_SUCCESS
	}}
	e, err := GraphInstantiate(d, 0x6A6A, 0)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if e != 0x7E7E {
		t.Errorf("exec = %#x, want 0x7E7E", e)
	}
}

func TestGraphLaunchAndDestroy(t *testing.T) {
	if err := GraphLaunch(nil, 0x7E7E, 0x5151); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("launch nil driver = %v, want ErrNotInitialized", err)
	}
	if err := GraphLaunch(&cudasys.Driver{}, 0x7E7E, 0x5151); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("launch nil func = %v, want ErrSymbolUnavailable", err)
	}
	var gotStream cudasys.CUstream
	d := &cudasys.Driver{
		CuGraphLaunch: func(_ cudasys.CUgraphExec, s cudasys.CUstream) cudasys.CUresult {
			gotStream = s
			return cudasys.CUDA_SUCCESS
		},
		CuGraphDestroy:     func(cudasys.CUgraph) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuGraphExecDestroy: func(cudasys.CUgraphExec) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
	}
	if err := GraphLaunch(d, 0x7E7E, 0x5151); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if gotStream != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", gotStream)
	}
	if err := GraphDestroy(d, 0x6A6A); err != nil {
		t.Errorf("graph destroy: %v", err)
	}
	if err := GraphExecDestroy(d, 0x7E7E); err != nil {
		t.Errorf("exec destroy: %v", err)
	}
	if err := GraphDestroy(nil, 0x6A6A); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("destroy nil driver = %v, want ErrNotInitialized", err)
	}
	if err := GraphDestroy(&cudasys.Driver{}, 0x6A6A); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("destroy nil func = %v, want ErrSymbolUnavailable", err)
	}
	if err := GraphExecDestroy(&cudasys.Driver{}, 0x7E7E); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("exec destroy nil func = %v, want ErrSymbolUnavailable", err)
	}
}

func TestGraphExecUpdate(t *testing.T) {
	if err := GraphExecUpdate(nil, 0x7E7E, 0x6A6A); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := GraphExecUpdate(&cudasys.Driver{}, 0x7E7E, 0x6A6A); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	ok := &cudasys.Driver{CuGraphExecUpdate: func(cudasys.CUgraphExec, cudasys.CUgraph, *cudasys.CUgraphNode, *int32) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}}
	if err := GraphExecUpdate(ok, 0x7E7E, 0x6A6A); err != nil {
		t.Errorf("update = %v, want nil", err)
	}
	fail := &cudasys.Driver{CuGraphExecUpdate: func(cudasys.CUgraphExec, cudasys.CUgraph, *cudasys.CUgraphNode, *int32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
	}}
	if err := GraphExecUpdate(fail, 0x7E7E, 0x6A6A); !errors.Is(err, ErrGraphExecUpdateFailure) {
		t.Errorf("update failure = %v, want ErrGraphExecUpdateFailure", err)
	}
}
