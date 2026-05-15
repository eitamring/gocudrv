package cudasys

import (
	"fmt"

	"github.com/ebitengine/purego"

	"github.com/eitamring/gocudrv/internal/dynload"
)

// Driver holds the bound CUDA driver function pointers and the underlying
// shared-library handle. Fields are public so tests can construct a fake
// Driver without touching purego.
type Driver struct {
	lib                       dynload.Library
	CuInit                    func(flags uint32) CUresult
	CuDriverGetVersion        func(version *int32) CUresult
	CuDeviceGetCount          func(count *int32) CUresult
	CuDeviceGet               func(device *CUdevice, ordinal int32) CUresult
	CuDeviceGetName           func(name *byte, length int32, dev CUdevice) CUresult
	CuDeviceTotalMem          func(bytes *uint64, dev CUdevice) CUresult
	CuDeviceGetAttribute      func(value *int32, attr int32, dev CUdevice) CUresult
	CuCtxGetCurrent           func(ctx *CUcontext) CUresult
	CuCtxSetCurrent           func(ctx CUcontext) CUresult
	CuCtxSynchronize          func() CUresult
	CuDevicePrimaryCtxRetain  func(ctx *CUcontext, dev CUdevice) CUresult
	CuDevicePrimaryCtxRelease func(dev CUdevice) CUresult
	CuMemAlloc                func(devPtr *CUdeviceptr, bytesize uint64) CUresult
	CuMemFree                 func(devPtr CUdeviceptr) CUresult
	CuMemcpyHtoD              func(dst CUdeviceptr, src *byte, byteCount uint64) CUresult
	CuMemcpyDtoH              func(dst *byte, src CUdeviceptr, byteCount uint64) CUresult
	CuMemAllocHost            func(pp **byte, bytesize uint64) CUresult
	CuMemFreeHost             func(p *byte) CUresult
	CuModuleLoadData          func(module *CUmodule, image *byte) CUresult
	CuModuleUnload            func(module CUmodule) CUresult
	CuModuleGetFunction       func(fn *CUfunction, module CUmodule, name *byte) CUresult
}

// bindFn is the symbol-binding function used by Load. Overridable in tests.
var bindFn = bind

// Load binds the v0 set of CUDA driver symbols from lib. If any binding
// fails, the library is closed before returning the error so callers do not
// have to track ownership of the handle on the failure path.
func Load(lib dynload.Library) (*Driver, error) {
	d := &Driver{lib: lib}
	binds := []struct {
		fn   any
		name string
	}{
		{&d.CuInit, "cuInit"},
		{&d.CuDriverGetVersion, "cuDriverGetVersion"},
		{&d.CuDeviceGetCount, "cuDeviceGetCount"},
		{&d.CuDeviceGet, "cuDeviceGet"},
		{&d.CuDeviceGetName, "cuDeviceGetName"},
		{&d.CuDeviceTotalMem, "cuDeviceTotalMem_v2"},
		{&d.CuDeviceGetAttribute, "cuDeviceGetAttribute"},
		{&d.CuCtxGetCurrent, "cuCtxGetCurrent"},
		{&d.CuCtxSetCurrent, "cuCtxSetCurrent"},
		{&d.CuCtxSynchronize, "cuCtxSynchronize"},
		{&d.CuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"},
		{&d.CuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"},
		{&d.CuMemAlloc, "cuMemAlloc_v2"},
		{&d.CuMemFree, "cuMemFree_v2"},
		{&d.CuMemcpyHtoD, "cuMemcpyHtoD_v2"},
		{&d.CuMemcpyDtoH, "cuMemcpyDtoH_v2"},
		{&d.CuMemAllocHost, "cuMemAllocHost_v2"},
		{&d.CuMemFreeHost, "cuMemFreeHost"},
		{&d.CuModuleLoadData, "cuModuleLoadData"},
		{&d.CuModuleUnload, "cuModuleUnload"},
		{&d.CuModuleGetFunction, "cuModuleGetFunction"},
	}
	for _, b := range binds {
		if err := bindFn(lib, b.fn, b.name); err != nil {
			_ = lib.Close()
			return nil, err
		}
	}
	return d, nil
}

// Close releases the underlying shared library, if any. Test-constructed
// Drivers without a library are a no-op. Safe to call on a nil receiver and
// safe to call more than once.
func (d *Driver) Close() error {
	if d == nil || d.lib == nil {
		return nil
	}
	lib := d.lib
	d.lib = nil
	return lib.Close()
}

func bind(lib dynload.Library, fn any, name string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("cudasys: bind %q: %v", name, r)
		}
	}()
	purego.RegisterLibFunc(fn, lib.Handle(), name)
	return nil
}
