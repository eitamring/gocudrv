// The dispatch tables below were generated once from the Driver struct's own
// signatures and are maintained by hand; see docs/internals.md.

package cudasys

import (
	"unsafe"

	"github.com/ebitengine/purego"
)

// cures converts a SyscallN result register into a CUresult.
func cures(r1, _, _ uintptr) CUresult { return CUresult(int32(r1)) }

// boundSymbol pairs a driver entry-point name with the assignment that binds
// its SyscallN adapter to the matching Driver field.
type boundSymbol struct {
	name string
	set  func(d *Driver, fnAddr uintptr)
}

var requiredSymbols = []boundSymbol{
	{"cuInit", func(d *Driver, fnAddr uintptr) {
		d.CuInit = func(flags uint32) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(flags))) }
	}},
	{"cuDriverGetVersion", func(d *Driver, fnAddr uintptr) {
		d.CuDriverGetVersion = func(version *int32) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(version)))) }
	}},
	{"cuDeviceGetCount", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGetCount = func(count *int32) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(count)))) }
	}},
	{"cuDeviceGet", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGet = func(device *CUdevice, ordinal int32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(device)), uintptr(ordinal)))
		}
	}},
	{"cuDeviceGetName", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGetName = func(name *byte, length int32, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(name)), uintptr(length), uintptr(dev)))
		}
	}},
	{"cuDeviceTotalMem_v2", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceTotalMem = func(bytes *uint64, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(bytes)), uintptr(dev)))
		}
	}},
	{"cuDeviceGetAttribute", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGetAttribute = func(value *int32, attr int32, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(value)), uintptr(attr), uintptr(dev)))
		}
	}},
	{"cuCtxGetCurrent", func(d *Driver, fnAddr uintptr) {
		d.CuCtxGetCurrent = func(ctx *CUcontext) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(ctx)))) }
	}},
	{"cuCtxSetCurrent", func(d *Driver, fnAddr uintptr) {
		d.CuCtxSetCurrent = func(ctx CUcontext) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(ctx))) }
	}},
	{"cuCtxSynchronize", func(d *Driver, fnAddr uintptr) {
		d.CuCtxSynchronize = func() CUresult { return cures(purego.SyscallN(fnAddr)) }
	}},
	{"cuCtxGetStreamPriorityRange", func(d *Driver, fnAddr uintptr) {
		d.CuCtxGetStreamPriorityRange = func(leastPriority *int32, greatestPriority *int32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(leastPriority)), uintptr(unsafe.Pointer(greatestPriority))))
		}
	}},
	{"cuDevicePrimaryCtxRetain", func(d *Driver, fnAddr uintptr) {
		d.CuDevicePrimaryCtxRetain = func(ctx *CUcontext, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(ctx)), uintptr(dev)))
		}
	}},
	{"cuDevicePrimaryCtxRelease_v2", func(d *Driver, fnAddr uintptr) {
		d.CuDevicePrimaryCtxRelease = func(dev CUdevice) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(dev))) }
	}},
	{"cuMemAlloc_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemAlloc = func(devPtr *CUdeviceptr, bytesize uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(devPtr)), uintptr(bytesize)))
		}
	}},
	{"cuMemFree_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemFree = func(devPtr CUdeviceptr) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(devPtr))) }
	}},
	{"cuMemGetInfo_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemGetInfo = func(free *uint64, total *uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(free)), uintptr(unsafe.Pointer(total))))
		}
	}},
	{"cuMemcpyHtoD_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyHtoD = func(dst CUdeviceptr, src *byte, byteCount uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(unsafe.Pointer(src)), uintptr(byteCount)))
		}
	}},
	{"cuMemcpyDtoH_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyDtoH = func(dst *byte, src CUdeviceptr, byteCount uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(dst)), uintptr(src), uintptr(byteCount)))
		}
	}},
	{"cuMemcpyDtoD_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyDtoD = func(dst CUdeviceptr, src CUdeviceptr, byteCount uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(src), uintptr(byteCount)))
		}
	}},
	{"cuMemcpyHtoDAsync_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyHtoDAsync = func(dst CUdeviceptr, src *byte, byteCount uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(unsafe.Pointer(src)), uintptr(byteCount), uintptr(stream)))
		}
	}},
	{"cuMemcpyDtoHAsync_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyDtoHAsync = func(dst *byte, src CUdeviceptr, byteCount uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(dst)), uintptr(src), uintptr(byteCount), uintptr(stream)))
		}
	}},
	{"cuMemcpyDtoDAsync_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyDtoDAsync = func(dst CUdeviceptr, src CUdeviceptr, byteCount uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(src), uintptr(byteCount), uintptr(stream)))
		}
	}},
	{"cuMemsetD8_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemsetD8 = func(dst CUdeviceptr, value uint8, count uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(value), uintptr(count)))
		}
	}},
	{"cuMemsetD16_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemsetD16 = func(dst CUdeviceptr, value uint16, count uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(value), uintptr(count)))
		}
	}},
	{"cuMemsetD32_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemsetD32 = func(dst CUdeviceptr, value uint32, count uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(value), uintptr(count)))
		}
	}},
	{"cuMemsetD8Async", func(d *Driver, fnAddr uintptr) {
		d.CuMemsetD8Async = func(dst CUdeviceptr, value uint8, count uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(value), uintptr(count), uintptr(stream)))
		}
	}},
	{"cuMemsetD16Async", func(d *Driver, fnAddr uintptr) {
		d.CuMemsetD16Async = func(dst CUdeviceptr, value uint16, count uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(value), uintptr(count), uintptr(stream)))
		}
	}},
	{"cuMemsetD32Async", func(d *Driver, fnAddr uintptr) {
		d.CuMemsetD32Async = func(dst CUdeviceptr, value uint32, count uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dst), uintptr(value), uintptr(count), uintptr(stream)))
		}
	}},
	{"cuMemAllocHost_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemAllocHost = func(pp **byte, bytesize uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pp)), uintptr(bytesize)))
		}
	}},
	{"cuMemFreeHost", func(d *Driver, fnAddr uintptr) {
		d.CuMemFreeHost = func(p *byte) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(p)))) }
	}},
	{"cuModuleLoadData", func(d *Driver, fnAddr uintptr) {
		d.CuModuleLoadData = func(module *CUmodule, image *byte) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(module)), uintptr(unsafe.Pointer(image))))
		}
	}},
	{"cuModuleLoadDataEx", func(d *Driver, fnAddr uintptr) {
		d.CuModuleLoadDataEx = func(module *CUmodule, image *byte, numOptions uint32, options *int32, optionValues *uintptr) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(module)), uintptr(unsafe.Pointer(image)), uintptr(numOptions), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(optionValues))))
		}
	}},
	{"cuModuleUnload", func(d *Driver, fnAddr uintptr) {
		d.CuModuleUnload = func(module CUmodule) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(module))) }
	}},
	{"cuModuleGetFunction", func(d *Driver, fnAddr uintptr) {
		d.CuModuleGetFunction = func(fn *CUfunction, module CUmodule, name *byte) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(fn)), uintptr(module), uintptr(unsafe.Pointer(name))))
		}
	}},
	{"cuModuleGetGlobal_v2", func(d *Driver, fnAddr uintptr) {
		d.CuModuleGetGlobal = func(dptr *CUdeviceptr, bytes *uint64, module CUmodule, name *byte) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(dptr)), uintptr(unsafe.Pointer(bytes)), uintptr(module), uintptr(unsafe.Pointer(name))))
		}
	}},
	{"cuStreamCreate", func(d *Driver, fnAddr uintptr) {
		d.CuStreamCreate = func(stream *CUstream, flags uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(stream)), uintptr(flags)))
		}
	}},
	{"cuStreamCreateWithPriority", func(d *Driver, fnAddr uintptr) {
		d.CuStreamCreateWithPriority = func(stream *CUstream, flags uint32, priority int32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(stream)), uintptr(flags), uintptr(priority)))
		}
	}},
	{"cuStreamDestroy_v2", func(d *Driver, fnAddr uintptr) {
		d.CuStreamDestroy = func(stream CUstream) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(stream))) }
	}},
	{"cuStreamSynchronize", func(d *Driver, fnAddr uintptr) {
		d.CuStreamSynchronize = func(stream CUstream) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(stream))) }
	}},
	{"cuStreamQuery", func(d *Driver, fnAddr uintptr) {
		d.CuStreamQuery = func(stream CUstream) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(stream))) }
	}},
	{"cuStreamWaitEvent", func(d *Driver, fnAddr uintptr) {
		d.CuStreamWaitEvent = func(stream CUstream, event CUevent, flags uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(stream), uintptr(event), uintptr(flags)))
		}
	}},
	{"cuEventCreate", func(d *Driver, fnAddr uintptr) {
		d.CuEventCreate = func(event *CUevent, flags uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(event)), uintptr(flags)))
		}
	}},
	{"cuEventDestroy_v2", func(d *Driver, fnAddr uintptr) {
		d.CuEventDestroy = func(event CUevent) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(event))) }
	}},
	{"cuEventRecord", func(d *Driver, fnAddr uintptr) {
		d.CuEventRecord = func(event CUevent, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(event), uintptr(stream)))
		}
	}},
	{"cuEventQuery", func(d *Driver, fnAddr uintptr) {
		d.CuEventQuery = func(event CUevent) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(event))) }
	}},
	{"cuEventSynchronize", func(d *Driver, fnAddr uintptr) {
		d.CuEventSynchronize = func(event CUevent) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(event))) }
	}},
	{"cuEventElapsedTime", func(d *Driver, fnAddr uintptr) {
		d.CuEventElapsedTime = func(ms *float32, start CUevent, end CUevent) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(ms)), uintptr(start), uintptr(end)))
		}
	}},
	{"cuLaunchKernel", func(d *Driver, fnAddr uintptr) {
		d.CuLaunchKernel = func(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes uint32, stream CUstream, kernelParams *unsafe.Pointer, extra *unsafe.Pointer) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(fn), uintptr(gridX), uintptr(gridY), uintptr(gridZ), uintptr(blockX), uintptr(blockY), uintptr(blockZ), uintptr(sharedMemBytes), uintptr(stream), uintptr(unsafe.Pointer(kernelParams)), uintptr(unsafe.Pointer(extra))))
		}
	}},
}

var optionalSymbols = []boundSymbol{
	{"cuMemAllocAsync", func(d *Driver, fnAddr uintptr) {
		d.CuMemAllocAsync = func(devPtr *CUdeviceptr, bytesize uint64, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(devPtr)), uintptr(bytesize), uintptr(stream)))
		}
	}},
	{"cuMemFreeAsync", func(d *Driver, fnAddr uintptr) {
		d.CuMemFreeAsync = func(devPtr CUdeviceptr, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(devPtr), uintptr(stream)))
		}
	}},
	{"cuOccupancyMaxActiveBlocksPerMultiprocessor", func(d *Driver, fnAddr uintptr) {
		d.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(numBlocks *int32, fn CUfunction, blockSize int32, dynamicSMemSize uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(numBlocks)), uintptr(fn), uintptr(blockSize), uintptr(dynamicSMemSize)))
		}
	}},
	{"cuOccupancyMaxPotentialBlockSize", func(d *Driver, fnAddr uintptr) {
		d.CuOccupancyMaxPotentialBlockSize = func(minGridSize *int32, blockSize *int32, fn CUfunction, blockSizeToDynamicSMemSize uintptr, dynamicSMemSize uint64, blockSizeLimit int32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(minGridSize)), uintptr(unsafe.Pointer(blockSize)), uintptr(fn), blockSizeToDynamicSMemSize, uintptr(dynamicSMemSize), uintptr(blockSizeLimit)))
		}
	}},
	{"cuStreamBeginCapture_v2", func(d *Driver, fnAddr uintptr) {
		d.CuStreamBeginCapture = func(stream CUstream, mode uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(stream), uintptr(mode)))
		}
	}},
	{"cuStreamEndCapture", func(d *Driver, fnAddr uintptr) {
		d.CuStreamEndCapture = func(stream CUstream, graph *CUgraph) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(stream), uintptr(unsafe.Pointer(graph))))
		}
	}},
	{"cuGraphInstantiateWithFlags", func(d *Driver, fnAddr uintptr) {
		d.CuGraphInstantiate = func(execGraph *CUgraphExec, graph CUgraph, flags uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(execGraph)), uintptr(graph), uintptr(flags)))
		}
	}},
	{"cuGraphLaunch", func(d *Driver, fnAddr uintptr) {
		d.CuGraphLaunch = func(execGraph CUgraphExec, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(execGraph), uintptr(stream)))
		}
	}},
	{"cuGraphDestroy", func(d *Driver, fnAddr uintptr) {
		d.CuGraphDestroy = func(graph CUgraph) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(graph))) }
	}},
	{"cuGraphExecDestroy", func(d *Driver, fnAddr uintptr) {
		d.CuGraphExecDestroy = func(execGraph CUgraphExec) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(execGraph))) }
	}},
	{"cuGraphExecUpdate", func(d *Driver, fnAddr uintptr) {
		d.CuGraphExecUpdate = func(execGraph CUgraphExec, graph CUgraph, errNode *CUgraphNode, updateResult *int32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(execGraph), uintptr(graph), uintptr(unsafe.Pointer(errNode)), uintptr(unsafe.Pointer(updateResult))))
		}
	}},
	{"cuDeviceGetPCIBusId", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGetPCIBusId = func(pciBusId *byte, length int32, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pciBusId)), uintptr(length), uintptr(dev)))
		}
	}},
	{"cuDeviceGetUuid", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGetUuid = func(uuid *byte, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(uuid)), uintptr(dev)))
		}
	}},
	{"cuMemHostRegister_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemHostRegister = func(p *byte, bytesize uint64, flags uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(p)), uintptr(bytesize), uintptr(flags)))
		}
	}},
	{"cuMemHostUnregister", func(d *Driver, fnAddr uintptr) {
		d.CuMemHostUnregister = func(p *byte) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(p)))) }
	}},
	{"cuMemAllocPitch_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemAllocPitch = func(dptr *CUdeviceptr, pitch *uint64, widthInBytes uint64, height uint64, elementSizeBytes uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(dptr)), uintptr(unsafe.Pointer(pitch)), uintptr(widthInBytes), uintptr(height), uintptr(elementSizeBytes)))
		}
	}},
	{"cuMemcpy2D_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpy2D = func(pCopy *Memcpy2D) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pCopy)))) }
	}},
	{"cuMemcpy2DAsync_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpy2DAsync = func(pCopy *Memcpy2D, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pCopy)), uintptr(stream)))
		}
	}},
	{"cuMemcpy3D_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpy3D = func(pCopy *Memcpy3D) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pCopy)))) }
	}},
	{"cuMemcpy3DAsync_v2", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpy3DAsync = func(pCopy *Memcpy3D, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pCopy)), uintptr(stream)))
		}
	}},
	{"cuArrayCreate_v2", func(d *Driver, fnAddr uintptr) {
		d.CuArrayCreate = func(pHandle *CUarray, pAllocateArray *CUDA_ARRAY_DESCRIPTOR) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pHandle)), uintptr(unsafe.Pointer(pAllocateArray))))
		}
	}},
	{"cuArray3DCreate_v2", func(d *Driver, fnAddr uintptr) {
		d.CuArray3DCreate = func(pHandle *CUarray, pAllocateArray *CUDA_ARRAY3D_DESCRIPTOR) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pHandle)), uintptr(unsafe.Pointer(pAllocateArray))))
		}
	}},
	{"cuArrayDestroy", func(d *Driver, fnAddr uintptr) {
		d.CuArrayDestroy = func(hArray CUarray) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(hArray))) }
	}},
	{"cuTexObjectCreate", func(d *Driver, fnAddr uintptr) {
		d.CuTexObjectCreate = func(pTexObject *CUtexObject, pResDesc *CUDA_RESOURCE_DESC, pTexDesc *CUDA_TEXTURE_DESC, pResViewDesc unsafe.Pointer) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pTexObject)), uintptr(unsafe.Pointer(pResDesc)), uintptr(unsafe.Pointer(pTexDesc)), uintptr(pResViewDesc)))
		}
	}},
	{"cuTexObjectDestroy", func(d *Driver, fnAddr uintptr) {
		d.CuTexObjectDestroy = func(texObject CUtexObject) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(texObject))) }
	}},
	{"cuSurfObjectCreate", func(d *Driver, fnAddr uintptr) {
		d.CuSurfObjectCreate = func(pSurfObject *CUsurfObject, pResDesc *CUDA_RESOURCE_DESC) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pSurfObject)), uintptr(unsafe.Pointer(pResDesc))))
		}
	}},
	{"cuSurfObjectDestroy", func(d *Driver, fnAddr uintptr) {
		d.CuSurfObjectDestroy = func(surfObject CUsurfObject) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(surfObject))) }
	}},
	{"cuDeviceGetDefaultMemPool", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceGetDefaultMemPool = func(pool *CUmemoryPool, dev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pool)), uintptr(dev)))
		}
	}},
	{"cuMemPoolGetAttribute", func(d *Driver, fnAddr uintptr) {
		d.CuMemPoolGetAttribute = func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(pool), uintptr(attr), uintptr(value)))
		}
	}},
	{"cuMemPoolSetAttribute", func(d *Driver, fnAddr uintptr) {
		d.CuMemPoolSetAttribute = func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(pool), uintptr(attr), uintptr(value)))
		}
	}},
	{"cuMemAllocFromPoolAsync", func(d *Driver, fnAddr uintptr) {
		d.CuMemAllocFromPoolAsync = func(dptr *CUdeviceptr, bytesize uint64, pool CUmemoryPool, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(dptr)), uintptr(bytesize), uintptr(pool), uintptr(stream)))
		}
	}},
	{"cuMemAllocManaged", func(d *Driver, fnAddr uintptr) {
		d.CuMemAllocManaged = func(pp **byte, bytesize uint64, flags uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pp)), uintptr(bytesize), uintptr(flags)))
		}
	}},
	{"cuMemPrefetchAsync", func(d *Driver, fnAddr uintptr) {
		d.CuMemPrefetchAsync = func(devPtr CUdeviceptr, count uint64, dstDevice CUdevice, stream CUstream) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(devPtr), uintptr(count), uintptr(dstDevice), uintptr(stream)))
		}
	}},
	{"cuMemAdvise", func(d *Driver, fnAddr uintptr) {
		d.CuMemAdvise = func(devPtr CUdeviceptr, count uint64, advice int32, device CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(devPtr), uintptr(count), uintptr(advice), uintptr(device)))
		}
	}},
	{"cuMemGetAllocationGranularity", func(d *Driver, fnAddr uintptr) {
		d.CuMemGetAllocationGranularity = func(granularity *uint64, prop *CUmemAllocationProp, option uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(granularity)), uintptr(unsafe.Pointer(prop)), uintptr(option)))
		}
	}},
	{"cuMemCreate", func(d *Driver, fnAddr uintptr) {
		d.CuMemCreate = func(handle *CUmemGenericAllocationHandle, size uint64, prop *CUmemAllocationProp, flags uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(handle)), uintptr(size), uintptr(unsafe.Pointer(prop)), uintptr(flags)))
		}
	}},
	{"cuMemAddressReserve", func(d *Driver, fnAddr uintptr) {
		d.CuMemAddressReserve = func(ptr *CUdeviceptr, size uint64, alignment uint64, addr CUdeviceptr, flags uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(ptr)), uintptr(size), uintptr(alignment), uintptr(addr), uintptr(flags)))
		}
	}},
	{"cuMemMap", func(d *Driver, fnAddr uintptr) {
		d.CuMemMap = func(ptr CUdeviceptr, size uint64, offset uint64, handle CUmemGenericAllocationHandle, flags uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(ptr), uintptr(size), uintptr(offset), uintptr(handle), uintptr(flags)))
		}
	}},
	{"cuMemSetAccess", func(d *Driver, fnAddr uintptr) {
		d.CuMemSetAccess = func(ptr CUdeviceptr, size uint64, desc *CUmemAccessDesc, count uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(ptr), uintptr(size), uintptr(unsafe.Pointer(desc)), uintptr(count)))
		}
	}},
	{"cuMemUnmap", func(d *Driver, fnAddr uintptr) {
		d.CuMemUnmap = func(ptr CUdeviceptr, size uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(ptr), uintptr(size)))
		}
	}},
	{"cuMemAddressFree", func(d *Driver, fnAddr uintptr) {
		d.CuMemAddressFree = func(ptr CUdeviceptr, size uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(ptr), uintptr(size)))
		}
	}},
	{"cuMemRelease", func(d *Driver, fnAddr uintptr) {
		d.CuMemRelease = func(handle CUmemGenericAllocationHandle) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(handle)))
		}
	}},
	{"cuFuncSetAttribute", func(d *Driver, fnAddr uintptr) {
		d.CuFuncSetAttribute = func(fn CUfunction, attrib int32, value int32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(fn), uintptr(attrib), uintptr(value)))
		}
	}},
	{"cuFuncGetAttribute", func(d *Driver, fnAddr uintptr) {
		d.CuFuncGetAttribute = func(value *int32, attrib int32, fn CUfunction) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(value)), uintptr(attrib), uintptr(fn)))
		}
	}},
	{"cuPointerGetAttribute", func(d *Driver, fnAddr uintptr) {
		d.CuPointerGetAttribute = func(data unsafe.Pointer, attribute int32, ptr CUdeviceptr) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(data), uintptr(attribute), uintptr(ptr)))
		}
	}},
	{"cuDeviceCanAccessPeer", func(d *Driver, fnAddr uintptr) {
		d.CuDeviceCanAccessPeer = func(canAccess *int32, dev CUdevice, peerDev CUdevice) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(canAccess)), uintptr(dev), uintptr(peerDev)))
		}
	}},
	{"cuCtxEnablePeerAccess", func(d *Driver, fnAddr uintptr) {
		d.CuCtxEnablePeerAccess = func(peerContext CUcontext, flags uint32) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(peerContext), uintptr(flags)))
		}
	}},
	{"cuCtxDisablePeerAccess", func(d *Driver, fnAddr uintptr) {
		d.CuCtxDisablePeerAccess = func(peerContext CUcontext) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(peerContext))) }
	}},
	{"cuMemcpyPeer", func(d *Driver, fnAddr uintptr) {
		d.CuMemcpyPeer = func(dstDevice CUdeviceptr, dstContext CUcontext, srcDevice CUdeviceptr, srcContext CUcontext, byteCount uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(dstDevice), uintptr(dstContext), uintptr(srcDevice), uintptr(srcContext), uintptr(byteCount)))
		}
	}},
	{"cuLaunchCooperativeKernel", func(d *Driver, fnAddr uintptr) {
		d.CuLaunchCooperativeKernel = func(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes uint32, stream CUstream, kernelParams *unsafe.Pointer) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(fn), uintptr(gridX), uintptr(gridY), uintptr(gridZ), uintptr(blockX), uintptr(blockY), uintptr(blockZ), uintptr(sharedMemBytes), uintptr(stream), uintptr(unsafe.Pointer(kernelParams))))
		}
	}},
	{"cuLaunchHostFunc", func(d *Driver, fnAddr uintptr) {
		d.CuLaunchHostFunc = func(stream CUstream, fn uintptr, userData uintptr) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(stream), fn, userData))
		}
	}},
	{"cuIpcGetMemHandle", func(d *Driver, fnAddr uintptr) {
		d.CuIpcGetMemHandle = func(pHandle *CUipcMemHandle, dptr CUdeviceptr) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pHandle)), uintptr(dptr)))
		}
	}},
	{"cuIpcCloseMemHandle", func(d *Driver, fnAddr uintptr) {
		d.CuIpcCloseMemHandle = func(dptr CUdeviceptr) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(dptr))) }
	}},
	{"cuIpcGetEventHandle", func(d *Driver, fnAddr uintptr) {
		d.CuIpcGetEventHandle = func(pHandle *CUipcEventHandle, event CUevent) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(unsafe.Pointer(pHandle)), uintptr(event)))
		}
	}},
	{"cuLinkCreate_v2", func(d *Driver, fnAddr uintptr) {
		d.CuLinkCreate = func(numOptions uint32, options *int32, optionValues *uintptr, stateOut *CUlinkState) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(numOptions), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(optionValues)), uintptr(unsafe.Pointer(stateOut))))
		}
	}},
	{"cuLinkAddData_v2", func(d *Driver, fnAddr uintptr) {
		d.CuLinkAddData = func(state CUlinkState, inputType uint32, data *byte, size uint64, name *byte, numOptions uint32, options *int32, optionValues *uintptr) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(state), uintptr(inputType), uintptr(unsafe.Pointer(data)), uintptr(size), uintptr(unsafe.Pointer(name)), uintptr(numOptions), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(optionValues))))
		}
	}},
	{"cuLinkComplete", func(d *Driver, fnAddr uintptr) {
		d.CuLinkComplete = func(state CUlinkState, cubinOut *unsafe.Pointer, sizeOut *uint64) CUresult {
			return cures(purego.SyscallN(fnAddr, uintptr(state), uintptr(unsafe.Pointer(cubinOut)), uintptr(unsafe.Pointer(sizeOut))))
		}
	}},
	{"cuLinkDestroy", func(d *Driver, fnAddr uintptr) {
		d.CuLinkDestroy = func(state CUlinkState) CUresult { return cures(purego.SyscallN(fnAddr, uintptr(state))) }
	}},
}
