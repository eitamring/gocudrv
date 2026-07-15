//go:build !race

package cuda

import (
	"context"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

var globalErrorSink error

func TestGlobalCopyAllocs(t *testing.T) {
	ctx := newTestContext(t, fakeMemoryDriver(&memCalls{}, testGlobalPtr))
	g := &Global{module: &Module{ctx: ctx}, ptr: testGlobalPtr, bytes: testGlobalBytes}
	vals := []float32{1, 2, 3, 4}
	bg := context.Background()
	if allocs := testing.AllocsPerRun(1000, func() { globalErrorSink = WriteGlobal(bg, g, vals) }); allocs != 0 {
		t.Errorf("WriteGlobal allocations = %v, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(1000, func() { globalErrorSink = ReadGlobal(bg, vals, g) }); allocs != 0 {
		t.Errorf("ReadGlobal allocations = %v, want 0", allocs)
	}
}

func TestEventQueryAllocs(t *testing.T) {
	driver := fakeEventDriver(&eventCalls{}, nil)
	driver.CuEventQuery = func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_ERROR_NOT_READY }
	ctx := newTestContext(t, driver)
	event := &Event{ctx: ctx, raw: 1}
	if allocs := testing.AllocsPerRun(1000, func() { eventErrorSink = event.Query() }); allocs != 0 {
		t.Errorf("Query allocations = %v, want 0", allocs)
	}
}
