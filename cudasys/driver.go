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
	CuMemcpy3D                  func(pCopy *Memcpy3D) CUresult
	CuMemcpy3DAsync             func(pCopy *Memcpy3D, stream CUstream) CUresult
	CuArrayCreate               func(pHandle *CUarray, pAllocateArray *CUDA_ARRAY_DESCRIPTOR) CUresult
	CuArray3DCreate             func(pHandle *CUarray, pAllocateArray *CUDA_ARRAY3D_DESCRIPTOR) CUresult
	CuArrayDestroy              func(hArray CUarray) CUresult
	CuTexObjectCreate           func(pTexObject *CUtexObject, pResDesc *CUDA_RESOURCE_DESC, pTexDesc *CUDA_TEXTURE_DESC, pResViewDesc unsafe.Pointer) CUresult
	CuTexObjectDestroy          func(texObject CUtexObject) CUresult
	CuSurfObjectCreate          func(pSurfObject *CUsurfObject, pResDesc *CUDA_RESOURCE_DESC) CUresult
	CuSurfObjectDestroy         func(surfObject CUsurfObject) CUresult
	CuDeviceGetDefaultMemPool   func(pool *CUmemoryPool, dev CUdevice) CUresult
	CuMemPoolGetAttribute       func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult
	CuMemPoolSetAttribute       func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult
	CuMemAllocFromPoolAsync     func(dptr *CUdeviceptr, bytesize uint64, pool CUmemoryPool, stream CUstream) CUresult
	CuModuleLoadData            func(module *CUmodule, image *byte) CUresult
	CuModuleLoadDataEx          func(module *CUmodule, image *byte, numOptions uint32, options *int32, optionValues *uintptr) CUresult
	CuModuleUnload              func(module CUmodule) CUresult
	CuModuleGetFunction         func(fn *CUfunction, module CUmodule, name *byte) CUresult
	CuModuleGetGlobal           func(dptr *CUdeviceptr, bytes *uint64, module CUmodule, name *byte) CUresult
	CuLinkCreate                func(numOptions uint32, options *int32, optionValues *uintptr, stateOut *CUlinkState) CUresult
	CuLinkAddData               func(state CUlinkState, inputType uint32, data *byte, size uint64, name *byte, numOptions uint32, options *int32, optionValues *uintptr) CUresult
	CuLinkComplete              func(state CUlinkState, cubinOut *unsafe.Pointer, sizeOut *uint64) CUresult
	CuLinkDestroy               func(state CUlinkState) CUresult
	CuStreamCreate              func(stream *CUstream, flags uint32) CUresult
	CuStreamCreateWithPriority  func(stream *CUstream, flags uint32, priority int32) CUresult
	CuStreamDestroy             func(stream CUstream) CUresult
	CuStreamSynchronize         func(stream CUstream) CUresult
	CuStreamQuery               func(stream CUstream) CUresult
	CuStreamWaitEvent           func(stream CUstream, event CUevent, flags uint32) CUresult
	CuEventCreate               func(event *CUevent, flags uint32) CUresult
	CuEventDestroy              func(event CUevent) CUresult
	CuEventRecord               func(event CUevent, stream CUstream) CUresult
	CuEventQuery                func(event CUevent) CUresult
	CuEventSynchronize          func(event CUevent) CUresult
	CuEventElapsedTime          func(ms *float32, start CUevent, end CUevent) CUresult
	CuLaunchKernel              func(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes uint32, stream CUstream, kernelParams *unsafe.Pointer, extra *unsafe.Pointer) CUresult
	CuLaunchCooperativeKernel   func(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes uint32, stream CUstream, kernelParams *unsafe.Pointer) CUresult
	CuLaunchHostFunc            func(stream CUstream, fn uintptr, userData uintptr) CUresult
	CuIpcGetMemHandle           func(pHandle *CUipcMemHandle, dptr CUdeviceptr) CUresult
	CuIpcOpenMemHandle          func(pdptr *CUdeviceptr, handle CUipcMemHandle, flags uint32) CUresult
	CuIpcCloseMemHandle         func(dptr CUdeviceptr) CUresult
	CuIpcGetEventHandle         func(pHandle *CUipcEventHandle, event CUevent) CUresult
	CuIpcOpenEventHandle        func(phEvent *CUevent, handle CUipcEventHandle) CUresult

	CuOccupancyMaxActiveBlocksPerMultiprocessor func(numBlocks *int32, fn CUfunction, blockSize int32, dynamicSMemSize uint64) CUresult
	CuOccupancyMaxPotentialBlockSize            func(minGridSize *int32, blockSize *int32, fn CUfunction, blockSizeToDynamicSMemSize uintptr, dynamicSMemSize uint64, blockSizeLimit int32) CUresult

	CuStreamBeginCapture func(stream CUstream, mode uint32) CUresult
	CuStreamEndCapture   func(stream CUstream, graph *CUgraph) CUresult
	CuGraphInstantiate   func(execGraph *CUgraphExec, graph CUgraph, flags uint64) CUresult
	CuGraphLaunch        func(execGraph CUgraphExec, stream CUstream) CUresult
	CuGraphDestroy       func(graph CUgraph) CUresult
	CuGraphExecDestroy   func(execGraph CUgraphExec) CUresult
	CuGraphExecUpdate    func(execGraph CUgraphExec, graph CUgraph, errNode *CUgraphNode, updateResult *int32) CUresult

	CuMemAllocManaged  func(pp **byte, bytesize uint64, flags uint32) CUresult
	CuMemPrefetchAsync func(devPtr CUdeviceptr, count uint64, dstDevice CUdevice, stream CUstream) CUresult
	CuMemAdvise        func(devPtr CUdeviceptr, count uint64, advice int32, device CUdevice) CUresult

	CuMemGetAllocationGranularity func(granularity *uint64, prop *CUmemAllocationProp, option uint32) CUresult
	CuMemCreate                   func(handle *CUmemGenericAllocationHandle, size uint64, prop *CUmemAllocationProp, flags uint64) CUresult
	CuMemAddressReserve           func(ptr *CUdeviceptr, size uint64, alignment uint64, addr CUdeviceptr, flags uint64) CUresult
	CuMemMap                      func(ptr CUdeviceptr, size uint64, offset uint64, handle CUmemGenericAllocationHandle, flags uint64) CUresult
	CuMemSetAccess                func(ptr CUdeviceptr, size uint64, desc *CUmemAccessDesc, count uint64) CUresult
	CuMemUnmap                    func(ptr CUdeviceptr, size uint64) CUresult
	CuMemAddressFree              func(ptr CUdeviceptr, size uint64) CUresult
	CuMemRelease                  func(handle CUmemGenericAllocationHandle) CUresult

	CuFuncSetAttribute     func(fn CUfunction, attrib int32, value int32) CUresult
	CuFuncGetAttribute     func(value *int32, attrib int32, fn CUfunction) CUresult
	CuPointerGetAttribute  func(data unsafe.Pointer, attribute int32, ptr CUdeviceptr) CUresult
	CuDeviceCanAccessPeer  func(canAccess *int32, dev CUdevice, peerDev CUdevice) CUresult
	CuCtxEnablePeerAccess  func(peerContext CUcontext, flags uint32) CUresult
	CuCtxDisablePeerAccess func(peerContext CUcontext) CUresult
	CuMemcpyPeer           func(dstDevice CUdeviceptr, dstContext CUcontext, srcDevice CUdeviceptr, srcContext CUcontext, byteCount uint64) CUresult
}

// lookupFn is the symbol-address lookup used by Load. Overridable in tests.
var lookupFn = lookupSymbol

// Load resolves each driver symbol to its raw address and assigns a SyscallN
// adapter to its Driver field. A missing required symbol fails Load and closes
// the library; feature symbols bind best-effort. See docs/internals.md.
func Load(lib dynload.Library) (*Driver, error) {
	d := &Driver{lib: lib}
	for _, s := range requiredSymbols {
		addr, err := lookupFn(lib, s.name)
		if err != nil {
			_ = lib.Close()
			return nil, err
		}
		s.set(d, addr)
	}
	for _, s := range optionalSymbols {
		if addr, err := lookupFn(lib, s.name); err == nil {
			s.set(d, addr)
		}
	}
	bindByValueIPC(lib, d)
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
