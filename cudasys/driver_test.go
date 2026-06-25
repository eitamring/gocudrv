package cudasys

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/internal/dynload"
)

type fakeLib struct {
	closed     int
	closeError error
}

func (f *fakeLib) Handle() uintptr { return 0 }
func (f *fakeLib) Close() error {
	f.closed++
	return f.closeError
}

func TestLoadClosesLibOnBindFailure(t *testing.T) {
	// A failure binding any required symbol must close the library and fail Load.
	for _, failOn := range expectedRequiredOrder {
		t.Run(failOn+" fails", func(t *testing.T) {
			prev := bindFn
			t.Cleanup(func() { bindFn = prev })
			bindFn = func(_ dynload.Library, _ any, name string) error {
				if name == failOn {
					return errors.New("bind: nope")
				}
				return nil
			}

			f := &fakeLib{}
			d, err := Load(f)
			if err == nil {
				t.Fatal("want error")
			}
			if d != nil {
				t.Error("want nil Driver on failure")
			}
			if f.closed != 1 {
				t.Errorf("closed = %d, want 1", f.closed)
			}
		})
	}
}

func TestLoadSuccessKeepsLib(t *testing.T) {
	prev := bindFn
	t.Cleanup(func() { bindFn = prev })
	bindFn = func(dynload.Library, any, string) error { return nil }

	f := &fakeLib{}
	d, err := Load(f)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if d == nil {
		t.Fatal("nil driver")
	}
	if f.closed != 0 {
		t.Errorf("closed = %d, want 0", f.closed)
	}

	if err := d.Close(); err != nil {
		t.Errorf("first close: %v", err)
	}
	if f.closed != 1 {
		t.Errorf("after first close = %d, want 1", f.closed)
	}
	if err := d.Close(); err != nil {
		t.Errorf("second close: %v", err)
	}
	if f.closed != 1 {
		t.Errorf("after second close = %d, want 1 (idempotent)", f.closed)
	}
}

func TestCloseOnNilReceiverAndEmptyDriver(t *testing.T) {
	var d *Driver
	if err := d.Close(); err != nil {
		t.Errorf("nil receiver: got %v, want nil", err)
	}
	empty := &Driver{}
	if err := empty.Close(); err != nil {
		t.Errorf("empty driver: got %v, want nil", err)
	}
}

// expectedRequiredOrder and expectedOptionalOrder are the exact sequences Load
// binds: required symbols first (fatal on failure), then optional feature
// symbols (best-effort). They mirror the tables in docs/internals.md; update
// both together when the bound surface changes.
var expectedRequiredOrder = []string{
	"cuInit",
	"cuDriverGetVersion",
	"cuDeviceGetCount",
	"cuDeviceGet",
	"cuDeviceGetName",
	"cuDeviceTotalMem_v2",
	"cuDeviceGetAttribute",
	"cuCtxGetCurrent",
	"cuCtxSetCurrent",
	"cuCtxSynchronize",
	"cuCtxGetStreamPriorityRange",
	"cuDevicePrimaryCtxRetain",
	"cuDevicePrimaryCtxRelease_v2",
	"cuMemAlloc_v2",
	"cuMemFree_v2",
	"cuMemGetInfo_v2",
	"cuMemcpyHtoD_v2",
	"cuMemcpyDtoH_v2",
	"cuMemcpyDtoD_v2",
	"cuMemcpyHtoDAsync_v2",
	"cuMemcpyDtoHAsync_v2",
	"cuMemcpyDtoDAsync_v2",
	"cuMemsetD8_v2",
	"cuMemsetD16_v2",
	"cuMemsetD32_v2",
	"cuMemsetD8Async",
	"cuMemsetD16Async",
	"cuMemsetD32Async",
	"cuMemAllocHost_v2",
	"cuMemFreeHost",
	"cuModuleLoadData",
	"cuModuleLoadDataEx",
	"cuModuleUnload",
	"cuModuleGetFunction",
	"cuModuleGetGlobal_v2",
	"cuStreamCreate",
	"cuStreamCreateWithPriority",
	"cuStreamDestroy_v2",
	"cuStreamSynchronize",
	"cuStreamQuery",
	"cuStreamWaitEvent",
	"cuEventCreate",
	"cuEventDestroy_v2",
	"cuEventRecord",
	"cuEventQuery",
	"cuEventSynchronize",
	"cuEventElapsedTime",
	"cuLaunchKernel",
}

var expectedOptionalOrder = []string{
	"cuMemAllocAsync",
	"cuMemFreeAsync",
	"cuOccupancyMaxActiveBlocksPerMultiprocessor",
	"cuOccupancyMaxPotentialBlockSize",
	"cuStreamBeginCapture_v2",
	"cuStreamEndCapture",
	"cuGraphInstantiateWithFlags",
	"cuGraphLaunch",
	"cuGraphDestroy",
	"cuGraphExecDestroy",
	"cuDeviceGetPCIBusId",
	"cuDeviceGetUuid",
	"cuMemHostRegister_v2",
	"cuMemHostUnregister",
	"cuMemAllocPitch_v2",
	"cuMemcpy2D_v2",
	"cuMemcpy2DAsync_v2",
	"cuDeviceGetDefaultMemPool",
	"cuMemPoolGetAttribute",
	"cuMemPoolSetAttribute",
	"cuMemAllocFromPoolAsync",
}

func TestLoadBindsExpectedSymbolsInOrder(t *testing.T) {
	prev := bindFn
	t.Cleanup(func() { bindFn = prev })

	var got []string
	seen := map[string]bool{}
	bindFn = func(_ dynload.Library, _ any, name string) error {
		if seen[name] {
			t.Errorf("symbol %q bound more than once", name)
		}
		seen[name] = true
		got = append(got, name)
		return nil
	}

	f := &fakeLib{}
	if _, err := Load(f); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	want := append(append([]string{}, expectedRequiredOrder...), expectedOptionalOrder...)
	if len(got) != len(want) {
		t.Fatalf("bound %d symbols, want %d", len(got), len(want))
	}
	for i, name := range want {
		if got[i] != name {
			t.Errorf("bind[%d] = %q, want %q", i, got[i], name)
		}
	}
}

func TestLoadStopsAtFirstBindFailure(t *testing.T) {
	const failAt = "cuMemGetInfo_v2"
	prev := bindFn
	t.Cleanup(func() { bindFn = prev })

	var attempted []string
	bindFn = func(_ dynload.Library, _ any, name string) error {
		attempted = append(attempted, name)
		if name == failAt {
			return errors.New("bind: nope")
		}
		return nil
	}

	f := &fakeLib{}
	d, err := Load(f)
	if err == nil {
		t.Fatal("want error")
	}
	if d != nil {
		t.Error("want nil Driver on failure")
	}
	if f.closed != 1 {
		t.Errorf("closed = %d, want 1", f.closed)
	}

	idx := -1
	for i, name := range expectedRequiredOrder {
		if name == failAt {
			idx = i
			break
		}
	}
	want := expectedRequiredOrder[:idx+1]
	if len(attempted) != len(want) {
		t.Fatalf("attempted %d binds before stopping, want %d", len(attempted), len(want))
	}
	for i, name := range want {
		if attempted[i] != name {
			t.Errorf("attempt[%d] = %q, want %q", i, attempted[i], name)
		}
	}
}

// TestLoadSkipsMissingOptionalSymbols checks that a driver missing an optional
// feature symbol still loads: the failing bind is swallowed and the library is
// kept open, so only calling the affected API surfaces the gap.
func TestLoadSkipsMissingOptionalSymbols(t *testing.T) {
	for _, missing := range expectedOptionalOrder {
		t.Run(missing+" missing", func(t *testing.T) {
			prev := bindFn
			t.Cleanup(func() { bindFn = prev })
			bindFn = func(_ dynload.Library, _ any, name string) error {
				if name == missing {
					return errors.New("bind: nope")
				}
				return nil
			}

			f := &fakeLib{}
			d, err := Load(f)
			if err != nil {
				t.Fatalf("Load failed on missing optional %q: %v", missing, err)
			}
			if d == nil {
				t.Fatal("want non-nil Driver")
			}
			if f.closed != 0 {
				t.Errorf("closed = %d, want 0 (library must stay open)", f.closed)
			}
		})
	}
}
