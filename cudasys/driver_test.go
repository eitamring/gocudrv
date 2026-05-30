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
	cases := []struct {
		name   string
		failOn string
	}{
		{"cuInit fails", "cuInit"},
		{"cuDriverGetVersion fails", "cuDriverGetVersion"},
		{"cuDeviceGetCount fails", "cuDeviceGetCount"},
		{"cuDeviceGet fails", "cuDeviceGet"},
		{"cuDeviceGetName fails", "cuDeviceGetName"},
		{"cuDeviceTotalMem_v2 fails", "cuDeviceTotalMem_v2"},
		{"cuDeviceGetAttribute fails", "cuDeviceGetAttribute"},
		{"cuCtxGetCurrent fails", "cuCtxGetCurrent"},
		{"cuCtxSetCurrent fails", "cuCtxSetCurrent"},
		{"cuCtxSynchronize fails", "cuCtxSynchronize"},
		{"cuCtxGetStreamPriorityRange fails", "cuCtxGetStreamPriorityRange"},
		{"cuDevicePrimaryCtxRetain fails", "cuDevicePrimaryCtxRetain"},
		{"cuDevicePrimaryCtxRelease_v2 fails", "cuDevicePrimaryCtxRelease_v2"},
		{"cuMemAlloc_v2 fails", "cuMemAlloc_v2"},
		{"cuMemFree_v2 fails", "cuMemFree_v2"},
		{"cuMemGetInfo_v2 fails", "cuMemGetInfo_v2"},
		{"cuMemcpyHtoD_v2 fails", "cuMemcpyHtoD_v2"},
		{"cuMemcpyDtoH_v2 fails", "cuMemcpyDtoH_v2"},
		{"cuMemcpyDtoD_v2 fails", "cuMemcpyDtoD_v2"},
		{"cuMemcpyHtoDAsync_v2 fails", "cuMemcpyHtoDAsync_v2"},
		{"cuMemcpyDtoHAsync_v2 fails", "cuMemcpyDtoHAsync_v2"},
		{"cuMemcpyDtoDAsync_v2 fails", "cuMemcpyDtoDAsync_v2"},
		{"cuMemsetD8_v2 fails", "cuMemsetD8_v2"},
		{"cuMemsetD32_v2 fails", "cuMemsetD32_v2"},
		{"cuMemsetD8Async fails", "cuMemsetD8Async"},
		{"cuMemsetD32Async fails", "cuMemsetD32Async"},
		{"cuMemAllocHost_v2 fails", "cuMemAllocHost_v2"},
		{"cuMemFreeHost fails", "cuMemFreeHost"},
		{"cuModuleLoadData fails", "cuModuleLoadData"},
		{"cuModuleUnload fails", "cuModuleUnload"},
		{"cuModuleGetFunction fails", "cuModuleGetFunction"},
		{"cuStreamCreate fails", "cuStreamCreate"},
		{"cuStreamCreateWithPriority fails", "cuStreamCreateWithPriority"},
		{"cuStreamDestroy_v2 fails", "cuStreamDestroy_v2"},
		{"cuStreamSynchronize fails", "cuStreamSynchronize"},
		{"cuStreamWaitEvent fails", "cuStreamWaitEvent"},
		{"cuEventCreate fails", "cuEventCreate"},
		{"cuEventDestroy_v2 fails", "cuEventDestroy_v2"},
		{"cuEventRecord fails", "cuEventRecord"},
		{"cuEventQuery fails", "cuEventQuery"},
		{"cuEventSynchronize fails", "cuEventSynchronize"},
		{"cuEventElapsedTime fails", "cuEventElapsedTime"},
		{"cuLaunchKernel fails", "cuLaunchKernel"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prev := bindFn
			t.Cleanup(func() { bindFn = prev })
			bindFn = func(_ dynload.Library, _ any, name string) error {
				if name == tc.failOn {
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
