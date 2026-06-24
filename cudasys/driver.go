package cudasys

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/eitamring/gocudrv/internal/dynload"
)

// Driver holds the bound CUDA driver function pointers and the underlying
// shared-library handle. Fields are public so tests can construct a fake
// Driver without touching purego.
type Driver struct {
	lib                         dynload.Library
	CuInit                      func(flags uint32) CUresult
	CuDriverGetVersion          func(version *int32) CUresult
	CuDeviceGetCount            func(count *int32) CUresult
	CuDeviceGet                 func(device *CUdevice, ordinal int32) CUresult
	CuDeviceGetName             func(name *byte, length int32, dev CUdevice) CUresult
	CuDeviceTotalMem            func(bytes *uint64, dev CUdevice) CUresult
	CuDeviceGetAttribute        func(value *int32, attr int32, dev CUdevice) CUresult
	CuDeviceGetPCIBusId         func(pciBusId *byte, length int32, dev CUdevice) CUresult
	CuDeviceGetUuid             func(uuid *byte, dev CUdevice) CUresult
	CuCtxGetCurrent             func(ctx *CUcontext) CUresult
	CuCtxSetCurrent             func(ctx CUcontext) CUresult
	CuCtxSynchronize            func() CUresult
	CuCtxGetStreamPriorityRange func(leastPriority *int32, greatestPriority *int32) CUresult
	CuDevicePrimaryCtxRetain    func(ctx *CUcontext, dev CUdevice) CUresult
	CuDevicePrimaryCtxRelease   func(dev CUdevice) CUresult
	CuMemAlloc                  func(devPtr *CUdeviceptr, bytesize uint64) CUresult
	CuMemFree                   func(devPtr CUdeviceptr) CUresult
	CuMemAllocAsync             func(devPtr *CUdeviceptr, bytesize uint64, stream CUstream) CUresult
	CuMemFreeAsync              func(devPtr CUdeviceptr, stream CUstream) CUresult
	CuMemGetInfo                func(free *uint64, total *uint64) CUresult
	CuMemcpyHtoD                func(dst CUdeviceptr, src *byte, byteCount uint64) CUresult
	CuMemcpyDtoH                func(dst *byte, src CUdeviceptr, byteCount uint64) CUresult
	CuMemcpyDtoD                func(dst CUdeviceptr, src CUdeviceptr, byteCount uint64) CUresult
	CuMemcpyHtoDAsync           func(dst CUdeviceptr, src *byte, byteCount uint64, stream CUstream) CUresult
	CuMemcpyDtoHAsync           func(dst *byte, src CUdeviceptr, byteCount uint64, stream CUstream) CUresult
	CuMemcpyDtoDAsync           func(dst CUdeviceptr, src CUdeviceptr, byteCount uint64, stream CUstream) CUresult
	CuMemsetD8                  func(dst CUdeviceptr, value uint8, count uint64) CUresult
	CuMemsetD16                 func(dst CUdeviceptr, value uint16, count uint64) CUresult
	CuMemsetD32                 func(dst CUdeviceptr, value uint32, count uint64) CUresult
	CuMemsetD8Async             func(dst CUdeviceptr, value uint8, count uint64, stream CUstream) CUresult
	CuMemsetD16Async            func(dst CUdeviceptr, value uint16, count uint64, stream CUstream) CUresult
	CuMemsetD32Async            func(dst CUdeviceptr, value uint32, count uint64, stream CUstream) CUresult
	CuMemAllocHost              func(pp **byte, bytesize uint64) CUresult
	CuMemFreeHost               func(p *byte) CUresult
	CuMemHostRegister           func(p *byte, bytesize uint64, flags uint32) CUresult
	CuMemHostUnregister         func(p *byte) CUresult
	CuMemAllocPitch             func(dptr *CUdeviceptr, pitch *uint64, widthInBytes uint64, height uint64, elementSizeBytes uint32) CUresult
	CuMemcpy2D                  func(pCopy *Memcpy2D) CUresult
	CuMemcpy2DAsync             func(pCopy *Memcpy2D, stream CUstream) CUresult
	CuDeviceGetDefaultMemPool   func(pool *CUmemoryPool, dev CUdevice) CUresult
	CuMemPoolGetAttribute       func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult
	CuMemPoolSetAttribute       func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult
	CuMemAllocFromPoolAsync     func(dptr *CUdeviceptr, bytesize uint64, pool CUmemoryPool, stream CUstream) CUresult
	CuModuleLoadData            func(module *CUmodule, image *byte) CUresult
	CuModuleUnload              func(module CUmodule) CUresult
	CuModuleGetFunction         func(fn *CUfunction, module CUmodule, name *byte) CUresult
	CuModuleGetGlobal           func(dptr *CUdeviceptr, bytes *uint64, module CUmodule, name *byte) CUresult
	CuStreamCreate              func(stream *CUstream, flags uint32) CUresult
	CuStreamCreateWithPriority  func(stream *CUstream, flags uint32, priority int32) CUresult
	CuStreamDestroy             func(stream CUstream) CUresult
	CuStreamSynchronize         func(stream CUstream) CUresult
	CuStreamWaitEvent           func(stream CUstream, event CUevent, flags uint32) CUresult
	CuEventCreate               func(event *CUevent, flags uint32) CUresult
	CuEventDestroy              func(event CUevent) CUresult
	CuEventRecord               func(event CUevent, stream CUstream) CUresult
	CuEventQuery                func(event CUevent) CUresult
	CuEventSynchronize          func(event CUevent) CUresult
	CuEventElapsedTime          func(ms *float32, start CUevent, end CUevent) CUresult
	CuLaunchKernel              func(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes uint32, stream CUstream, kernelParams *unsafe.Pointer, extra *unsafe.Pointer) CUresult

	CuOccupancyMaxActiveBlocksPerMultiprocessor func(numBlocks *int32, fn CUfunction, blockSize int32, dynamicSMemSize uint64) CUresult
	CuOccupancyMaxPotentialBlockSize            func(minGridSize *int32, blockSize *int32, fn CUfunction, blockSizeToDynamicSMemSize uintptr, dynamicSMemSize uint64, blockSizeLimit int32) CUresult

	CuStreamBeginCapture func(stream CUstream, mode uint32) CUresult
	CuStreamEndCapture   func(stream CUstream, graph *CUgraph) CUresult
	CuGraphInstantiate   func(execGraph *CUgraphExec, graph CUgraph, flags uint64) CUresult
	CuGraphLaunch        func(execGraph CUgraphExec, stream CUstream) CUresult
	CuGraphDestroy       func(graph CUgraph) CUresult
	CuGraphExecDestroy   func(execGraph CUgraphExec) CUresult
}

// bindFn is the symbol-binding function used by Load. Overridable in tests.
var bindFn = bind

// symbol pairs a destination function pointer with its driver entry-point name.
type symbol struct {
	fn   any
	name string
}

// Load binds the CUDA driver symbols from lib. Core symbols are required: if any
// of them fails to bind, the library is closed and the error is returned, so
// callers do not have to track ownership of the handle on the failure path.
// Feature symbols (async allocation, occupancy, graphs) are bound best-effort: a
// driver that does not export one leaves its pointer nil and the corresponding
// API returns ErrSymbolUnavailable when called, so newer features do not raise
// the minimum driver version for callers that never use them. See
// docs/internals.md for the symbol table and the practical minimum version.
func Load(lib dynload.Library) (*Driver, error) {
	d := &Driver{lib: lib}
	required := []symbol{
		// init
		{&d.CuInit, "cuInit"},
		{&d.CuDriverGetVersion, "cuDriverGetVersion"},
		// device discovery and attributes
		{&d.CuDeviceGetCount, "cuDeviceGetCount"},
		{&d.CuDeviceGet, "cuDeviceGet"},
		{&d.CuDeviceGetName, "cuDeviceGetName"},
		{&d.CuDeviceTotalMem, "cuDeviceTotalMem_v2"},
		{&d.CuDeviceGetAttribute, "cuDeviceGetAttribute"},
		// context and primary context
		{&d.CuCtxGetCurrent, "cuCtxGetCurrent"},
		{&d.CuCtxSetCurrent, "cuCtxSetCurrent"},
		{&d.CuCtxSynchronize, "cuCtxSynchronize"},
		{&d.CuCtxGetStreamPriorityRange, "cuCtxGetStreamPriorityRange"},
		{&d.CuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"},
		{&d.CuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"},
		// synchronous device and pinned host memory
		{&d.CuMemAlloc, "cuMemAlloc_v2"},
		{&d.CuMemFree, "cuMemFree_v2"},
		{&d.CuMemGetInfo, "cuMemGetInfo_v2"},
		{&d.CuMemcpyHtoD, "cuMemcpyHtoD_v2"},
		{&d.CuMemcpyDtoH, "cuMemcpyDtoH_v2"},
		{&d.CuMemcpyDtoD, "cuMemcpyDtoD_v2"},
		{&d.CuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2"},
		{&d.CuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2"},
		{&d.CuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2"},
		{&d.CuMemsetD8, "cuMemsetD8_v2"},
		{&d.CuMemsetD16, "cuMemsetD16_v2"},
		{&d.CuMemsetD32, "cuMemsetD32_v2"},
		{&d.CuMemsetD8Async, "cuMemsetD8Async"},
		{&d.CuMemsetD16Async, "cuMemsetD16Async"},
		{&d.CuMemsetD32Async, "cuMemsetD32Async"},
		{&d.CuMemAllocHost, "cuMemAllocHost_v2"},
		{&d.CuMemFreeHost, "cuMemFreeHost"},
		// module loading and globals
		{&d.CuModuleLoadData, "cuModuleLoadData"},
		{&d.CuModuleUnload, "cuModuleUnload"},
		{&d.CuModuleGetFunction, "cuModuleGetFunction"},
		{&d.CuModuleGetGlobal, "cuModuleGetGlobal_v2"},
		// streams
		{&d.CuStreamCreate, "cuStreamCreate"},
		{&d.CuStreamCreateWithPriority, "cuStreamCreateWithPriority"},
		{&d.CuStreamDestroy, "cuStreamDestroy_v2"},
		{&d.CuStreamSynchronize, "cuStreamSynchronize"},
		{&d.CuStreamWaitEvent, "cuStreamWaitEvent"},
		// events
		{&d.CuEventCreate, "cuEventCreate"},
		{&d.CuEventDestroy, "cuEventDestroy_v2"},
		{&d.CuEventRecord, "cuEventRecord"},
		{&d.CuEventQuery, "cuEventQuery"},
		{&d.CuEventSynchronize, "cuEventSynchronize"},
		{&d.CuEventElapsedTime, "cuEventElapsedTime"},
		// kernel launch
		{&d.CuLaunchKernel, "cuLaunchKernel"},
	}
	optional := []symbol{
		// stream-ordered async allocation (CUDA 11.2+)
		{&d.CuMemAllocAsync, "cuMemAllocAsync"},
		{&d.CuMemFreeAsync, "cuMemFreeAsync"},
		// occupancy helpers (CUDA 6.5+)
		{&d.CuOccupancyMaxActiveBlocksPerMultiprocessor, "cuOccupancyMaxActiveBlocksPerMultiprocessor"},
		{&d.CuOccupancyMaxPotentialBlockSize, "cuOccupancyMaxPotentialBlockSize"},
		// graph capture and replay (CUDA 11.x)
		{&d.CuStreamBeginCapture, "cuStreamBeginCapture_v2"},
		{&d.CuStreamEndCapture, "cuStreamEndCapture"},
		{&d.CuGraphInstantiate, "cuGraphInstantiateWithFlags"},
		{&d.CuGraphLaunch, "cuGraphLaunch"},
		{&d.CuGraphDestroy, "cuGraphDestroy"},
		{&d.CuGraphExecDestroy, "cuGraphExecDestroy"},
		// device diagnostics (PCI bus id CUDA 4.1+, uuid CUDA 9.2+)
		{&d.CuDeviceGetPCIBusId, "cuDeviceGetPCIBusId"},
		{&d.CuDeviceGetUuid, "cuDeviceGetUuid"},
		// host memory registration (CUDA 6.5+)
		{&d.CuMemHostRegister, "cuMemHostRegister_v2"},
		{&d.CuMemHostUnregister, "cuMemHostUnregister"},
		// pitched allocation and 2D copies (CUDA 3.2+)
		{&d.CuMemAllocPitch, "cuMemAllocPitch_v2"},
		{&d.CuMemcpy2D, "cuMemcpy2D_v2"},
		{&d.CuMemcpy2DAsync, "cuMemcpy2DAsync_v2"},
		// memory pools (CUDA 11.2+)
		{&d.CuDeviceGetDefaultMemPool, "cuDeviceGetDefaultMemPool"},
		{&d.CuMemPoolGetAttribute, "cuMemPoolGetAttribute"},
		{&d.CuMemPoolSetAttribute, "cuMemPoolSetAttribute"},
		{&d.CuMemAllocFromPoolAsync, "cuMemAllocFromPoolAsync"},
	}
	for _, b := range required {
		if err := bindFn(lib, b.fn, b.name); err != nil {
			_ = lib.Close()
			return nil, err
		}
	}
	for _, b := range optional {
		// Best-effort: a missing optional symbol leaves the pointer nil. The
		// wrapper for that call reports ErrSymbolUnavailable.
		_ = bindFn(lib, b.fn, b.name)
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
