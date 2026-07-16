//go:build !race

package cuda

import (
	"context"
	"runtime"
	"runtime/debug"
	"testing"
	"unsafe"

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

var packedSetErrorSink error

func TestLaunchPackedSetAllocs(t *testing.T) {
	var l launchFake
	drv := l.driver(t)
	drv.CuLaunchKernel = func(cudasys.CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, cudasys.CUstream, *unsafe.Pointer, *unsafe.Pointer) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	installDriver(t, drv)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	mod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("k")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	buf, err := Alloc[float32](ctx, 8)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	cfg := LaunchConfig1D(8, 8)
	bg := context.Background()

	small, err := Pack(Arg(buf), ArgValue(int32(0)))
	if err != nil {
		t.Fatalf("Pack small: %v", err)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		packedSetErrorSink = SetPacked(small, 1, int32(3))
		packedSetErrorSink = fn.LaunchPacked(bg, cfg, small)
	}); allocs != 0 {
		t.Errorf("small packed set allocs = %v, want 0", allocs)
	}

	args := make([]KernelArg, 20)
	for i := range args {
		args[i] = ArgValue(int32(i))
	}
	many, err := Pack(args...)
	if err != nil {
		t.Fatalf("Pack many: %v", err)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		packedSetErrorSink = SetPacked(many, 19, int32(7))
		packedSetErrorSink = fn.LaunchPacked(bg, cfg, many)
	}); allocs != 0 {
		t.Errorf("many packed set allocs = %v, want 0", allocs)
	}

	rawVal := [2]uint64{1, 2}
	raw, err := Pack(Arg(buf), ArgRaw(unsafe.Pointer(&rawVal), 16))
	if err != nil {
		t.Fatalf("Pack raw: %v", err)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		packedSetErrorSink = raw.SetRaw(1, unsafe.Pointer(&rawVal), 16)
		packedSetErrorSink = fn.LaunchPacked(bg, cfg, raw)
	}); allocs != 0 {
		t.Errorf("raw packed set allocs = %v, want 0", allocs)
	}

	spillArgs := make([]KernelArg, 20)
	for i := range spillArgs {
		spillArgs[i] = ArgValue(int32(i))
	}
	gcPrev := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(gcPrev)
	zeroSeen := false
	for try := 0; try < 3 && !zeroSeen; try++ {
		first, err := Pack(spillArgs...)
		if err != nil {
			t.Fatalf("Pack first: %v", err)
		}
		var m0, m1 runtime.MemStats
		runtime.ReadMemStats(&m0)
		packedSetErrorSink = fn.LaunchPacked(bg, cfg, first)
		runtime.ReadMemStats(&m1)
		zeroSeen = m1.Mallocs == m0.Mallocs
	}
	if !zeroSeen {
		t.Error("every first spilled LaunchPacked allocated, want a zero-alloc first launch")
	}
}
