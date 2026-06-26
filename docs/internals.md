# internals

The implementation is split into layers so public API code stays separate from
raw ABI details.

```text
public API             result wrapper           raw CUDA ABI             OS loader
----------             --------------           ------------             ---------
cuda.Init()       ->   cudaresult.Init()   ->    cudasys.Driver.CuInit
cuda.DeviceCount()     cudaresult.DeviceCount    cudasys.Driver.CuDeviceGetCount
cuda.GetDevice()       cudaresult.GetDevice      cudasys.Driver.CuDeviceGet
                                                  cudasys.Driver.CuDeviceGetName
                                                  cudasys.Driver.CuDeviceTotalMem
                                                  cudasys.Driver.CuDeviceGetAttribute
                                                        ^
                                                        |
                                             cudasys.Load(lib)
                                                        ^
                                                        |
                                      dynload.OpenAny(platform candidates)
```

## dynamic loading

`internal/platform.LibraryCandidates` returns CUDA driver library candidates by
OS.

| OS | candidates |
| --- | --- |
| linux | `libcuda.so.1`, `/usr/lib/x86_64-linux-gnu/libcuda.so.1`, `/usr/lib/wsl/lib/libcuda.so.1` |
| windows | `nvcuda.dll` |
| other | nil |

`internal/dynload.OpenAny` tries each candidate in order and returns the first
opened library. If every candidate fails, the returned error joins all failed
open attempts with their paths.

## raw bindings

`cudasys.Driver` stores bound CUDA driver functions and owns the library handle.

```go
type Driver struct {
    CuInit                    func(flags uint32) CUresult
    CuDriverGetVersion        func(version *int32) CUresult
    CuDeviceGetCount          func(count *int32) CUresult
    CuDeviceGet               func(device *CUdevice, ordinal int32) CUresult
    CuDeviceGetName           func(name *byte, length int32, dev CUdevice) CUresult
    CuDeviceTotalMem          func(bytes *uint64, dev CUdevice) CUresult
    CuDeviceGetAttribute      func(value *int32, attr int32, dev CUdevice) CUresult
    CuDeviceGetPCIBusId       func(pciBusId *byte, length int32, dev CUdevice) CUresult
    CuDeviceGetUuid           func(uuid *byte, dev CUdevice) CUresult
    CuCtxGetCurrent           func(ctx *CUcontext) CUresult
    CuCtxSetCurrent           func(ctx CUcontext) CUresult
    CuCtxSynchronize          func() CUresult
    CuCtxGetStreamPriorityRange func(leastPriority *int32, greatestPriority *int32) CUresult
    CuDevicePrimaryCtxRetain  func(ctx *CUcontext, dev CUdevice) CUresult
    CuDevicePrimaryCtxRelease func(dev CUdevice) CUresult
    CuMemAlloc                func(devPtr *CUdeviceptr, bytesize uint64) CUresult
    CuMemFree                 func(devPtr CUdeviceptr) CUresult
    CuMemAllocAsync           func(devPtr *CUdeviceptr, bytesize uint64, stream CUstream) CUresult
    CuMemFreeAsync            func(devPtr CUdeviceptr, stream CUstream) CUresult
    CuMemGetInfo              func(free *uint64, total *uint64) CUresult
    CuMemcpyHtoD              func(dst CUdeviceptr, src *byte, byteCount uint64) CUresult
    CuMemcpyDtoH              func(dst *byte, src CUdeviceptr, byteCount uint64) CUresult
    CuMemcpyDtoD              func(dst CUdeviceptr, src CUdeviceptr, byteCount uint64) CUresult
    CuMemcpyHtoDAsync         func(dst CUdeviceptr, src *byte, byteCount uint64, stream CUstream) CUresult
    CuMemcpyDtoHAsync         func(dst *byte, src CUdeviceptr, byteCount uint64, stream CUstream) CUresult
    CuMemcpyDtoDAsync         func(dst CUdeviceptr, src CUdeviceptr, byteCount uint64, stream CUstream) CUresult
    CuMemsetD8                func(dst CUdeviceptr, value uint8, count uint64) CUresult
    CuMemsetD16               func(dst CUdeviceptr, value uint16, count uint64) CUresult
    CuMemsetD32               func(dst CUdeviceptr, value uint32, count uint64) CUresult
    CuMemsetD8Async           func(dst CUdeviceptr, value uint8, count uint64, stream CUstream) CUresult
    CuMemsetD16Async          func(dst CUdeviceptr, value uint16, count uint64, stream CUstream) CUresult
    CuMemsetD32Async          func(dst CUdeviceptr, value uint32, count uint64, stream CUstream) CUresult
    CuMemAllocHost            func(pp **byte, bytesize uint64) CUresult
    CuMemFreeHost             func(p *byte) CUresult
    CuMemHostRegister         func(p *byte, bytesize uint64, flags uint32) CUresult
    CuMemHostUnregister       func(p *byte) CUresult
    CuMemAllocPitch           func(dptr *CUdeviceptr, pitch *uint64, widthInBytes, height uint64, elementSizeBytes uint32) CUresult
    CuMemcpy2D                func(pCopy *Memcpy2D) CUresult
    CuMemcpy2DAsync           func(pCopy *Memcpy2D, stream CUstream) CUresult
    CuDeviceGetDefaultMemPool func(pool *CUmemoryPool, dev CUdevice) CUresult
    CuMemPoolGetAttribute     func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult
    CuMemPoolSetAttribute     func(pool CUmemoryPool, attr int32, value unsafe.Pointer) CUresult
    CuMemAllocFromPoolAsync   func(dptr *CUdeviceptr, bytesize uint64, pool CUmemoryPool, stream CUstream) CUresult
    CuModuleLoadData          func(module *CUmodule, image *byte) CUresult
    CuModuleLoadDataEx        func(module *CUmodule, image *byte, numOptions uint32, options *int32, optionValues *uintptr) CUresult
    CuModuleUnload            func(module CUmodule) CUresult
    CuModuleGetFunction       func(fn *CUfunction, module CUmodule, name *byte) CUresult
    CuModuleGetGlobal         func(dptr *CUdeviceptr, bytes *uint64, module CUmodule, name *byte) CUresult
    CuStreamCreate            func(stream *CUstream, flags uint32) CUresult
    CuStreamCreateWithPriority func(stream *CUstream, flags uint32, priority int32) CUresult
    CuStreamDestroy           func(stream CUstream) CUresult
    CuStreamSynchronize       func(stream CUstream) CUresult
    CuStreamQuery             func(stream CUstream) CUresult
    CuStreamWaitEvent         func(stream CUstream, event CUevent, flags uint32) CUresult
    CuEventCreate             func(event *CUevent, flags uint32) CUresult
    CuEventDestroy            func(event CUevent) CUresult
    CuEventRecord             func(event CUevent, stream CUstream) CUresult
    CuEventQuery              func(event CUevent) CUresult
    CuEventSynchronize        func(event CUevent) CUresult
    CuEventElapsedTime        func(ms *float32, start CUevent, end CUevent) CUresult
    CuLaunchKernel            func(fn CUfunction, ..., kernelParams *unsafe.Pointer, extra *unsafe.Pointer) CUresult
    CuOccupancyMaxActiveBlocksPerMultiprocessor func(numBlocks *int32, fn CUfunction, blockSize int32, dynamicSMemSize uint64) CUresult
    CuOccupancyMaxPotentialBlockSize            func(minGridSize *int32, blockSize *int32, fn CUfunction, b2dSize uintptr, dynamicSMemSize uint64, blockSizeLimit int32) CUresult
    CuStreamBeginCapture func(stream CUstream, mode uint32) CUresult
    CuStreamEndCapture   func(stream CUstream, graph *CUgraph) CUresult
    CuGraphInstantiate   func(execGraph *CUgraphExec, graph CUgraph, flags uint64) CUresult
    CuGraphLaunch        func(execGraph CUgraphExec, stream CUstream) CUresult
    CuGraphDestroy       func(graph CUgraph) CUresult
    CuGraphExecDestroy   func(execGraph CUgraphExec) CUresult
    CuGraphExecUpdate    func(execGraph CUgraphExec, graph CUgraph, errNode *CUgraphNode, updateResult *int32) CUresult
}
```

`cudasys.Load` binds two groups of symbols. The core set is required: if any
core symbol is missing the load fails and the library is closed. The feature
groups (async allocation, occupancy, graphs) are bound best-effort: a driver
that does not export one still loads, leaves that function pointer nil, and the
wrapper for the affected call returns `ErrSymbolUnavailable`. This keeps newer
features from raising the minimum driver version for callers that never use
them.

Core symbols (always required at init):

| C entry point | `Driver` field | group |
| --- | --- | --- |
| `cuInit` | `CuInit` | init |
| `cuDriverGetVersion` | `CuDriverGetVersion` | init |
| `cuDeviceGetCount` | `CuDeviceGetCount` | device |
| `cuDeviceGet` | `CuDeviceGet` | device |
| `cuDeviceGetName` | `CuDeviceGetName` | device |
| `cuDeviceTotalMem_v2` | `CuDeviceTotalMem` | device |
| `cuDeviceGetAttribute` | `CuDeviceGetAttribute` | device |
| `cuCtxGetCurrent` | `CuCtxGetCurrent` | context |
| `cuCtxSetCurrent` | `CuCtxSetCurrent` | context |
| `cuCtxSynchronize` | `CuCtxSynchronize` | context |
| `cuCtxGetStreamPriorityRange` | `CuCtxGetStreamPriorityRange` | context |
| `cuDevicePrimaryCtxRetain` | `CuDevicePrimaryCtxRetain` | context |
| `cuDevicePrimaryCtxRelease_v2` | `CuDevicePrimaryCtxRelease` | context |
| `cuMemAlloc_v2` | `CuMemAlloc` | memory |
| `cuMemFree_v2` | `CuMemFree` | memory |
| `cuMemGetInfo_v2` | `CuMemGetInfo` | memory |
| `cuMemcpyHtoD_v2` | `CuMemcpyHtoD` | memory |
| `cuMemcpyDtoH_v2` | `CuMemcpyDtoH` | memory |
| `cuMemcpyDtoD_v2` | `CuMemcpyDtoD` | memory |
| `cuMemcpyHtoDAsync_v2` | `CuMemcpyHtoDAsync` | memory |
| `cuMemcpyDtoHAsync_v2` | `CuMemcpyDtoHAsync` | memory |
| `cuMemcpyDtoDAsync_v2` | `CuMemcpyDtoDAsync` | memory |
| `cuMemsetD8_v2` | `CuMemsetD8` | memory |
| `cuMemsetD16_v2` | `CuMemsetD16` | memory |
| `cuMemsetD32_v2` | `CuMemsetD32` | memory |
| `cuMemsetD8Async` | `CuMemsetD8Async` | memory |
| `cuMemsetD16Async` | `CuMemsetD16Async` | memory |
| `cuMemsetD32Async` | `CuMemsetD32Async` | memory |
| `cuMemAllocHost_v2` | `CuMemAllocHost` | memory |
| `cuMemFreeHost` | `CuMemFreeHost` | memory |
| `cuModuleLoadData` | `CuModuleLoadData` | module |
| `cuModuleLoadDataEx` | `CuModuleLoadDataEx` | module |
| `cuModuleUnload` | `CuModuleUnload` | module |
| `cuModuleGetFunction` | `CuModuleGetFunction` | module |
| `cuModuleGetGlobal_v2` | `CuModuleGetGlobal` | module |
| `cuStreamCreate` | `CuStreamCreate` | stream |
| `cuStreamCreateWithPriority` | `CuStreamCreateWithPriority` | stream |
| `cuStreamDestroy_v2` | `CuStreamDestroy` | stream |
| `cuStreamSynchronize` | `CuStreamSynchronize` | stream |
| `cuStreamQuery` | `CuStreamQuery` | stream |
| `cuStreamWaitEvent` | `CuStreamWaitEvent` | stream |
| `cuEventCreate` | `CuEventCreate` | event |
| `cuEventDestroy_v2` | `CuEventDestroy` | event |
| `cuEventRecord` | `CuEventRecord` | event |
| `cuEventQuery` | `CuEventQuery` | event |
| `cuEventSynchronize` | `CuEventSynchronize` | event |
| `cuEventElapsedTime` | `CuEventElapsedTime` | event |
| `cuLaunchKernel` | `CuLaunchKernel` | launch |

Feature symbols (bound best-effort; calls return `ErrSymbolUnavailable` if the
driver lacks the symbol):

| C entry point | `Driver` field | group | since |
| --- | --- | --- | --- |
| `cuMemAllocAsync` | `CuMemAllocAsync` | async allocation | CUDA 11.2 |
| `cuMemFreeAsync` | `CuMemFreeAsync` | async allocation | CUDA 11.2 |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | `CuOccupancyMaxActiveBlocksPerMultiprocessor` | occupancy | CUDA 6.5 |
| `cuOccupancyMaxPotentialBlockSize` | `CuOccupancyMaxPotentialBlockSize` | occupancy | CUDA 6.5 |
| `cuStreamBeginCapture_v2` | `CuStreamBeginCapture` | graph | CUDA 11.x |
| `cuStreamEndCapture` | `CuStreamEndCapture` | graph | CUDA 11.x |
| `cuGraphInstantiateWithFlags` | `CuGraphInstantiate` | graph | CUDA 11.x |
| `cuGraphLaunch` | `CuGraphLaunch` | graph | CUDA 11.x |
| `cuGraphDestroy` | `CuGraphDestroy` | graph | CUDA 11.x |
| `cuGraphExecDestroy` | `CuGraphExecDestroy` | graph | CUDA 11.x |
| `cuGraphExecUpdate` | `CuGraphExecUpdate` | graph | CUDA 10.2 |
| `cuDeviceGetPCIBusId` | `CuDeviceGetPCIBusId` | device diagnostics | CUDA 4.1 |
| `cuDeviceGetUuid` | `CuDeviceGetUuid` | device diagnostics | CUDA 9.2 |
| `cuMemHostRegister_v2` | `CuMemHostRegister` | host registration | CUDA 6.5 |
| `cuMemHostUnregister` | `CuMemHostUnregister` | host registration | CUDA 6.5 |
| `cuMemAllocPitch_v2` | `CuMemAllocPitch` | pitched memory | CUDA 3.2 |
| `cuMemcpy2D_v2` | `CuMemcpy2D` | pitched memory | CUDA 3.2 |
| `cuMemcpy2DAsync_v2` | `CuMemcpy2DAsync` | pitched memory | CUDA 3.2 |
| `cuDeviceGetDefaultMemPool` | `CuDeviceGetDefaultMemPool` | memory pools | CUDA 11.2 |
| `cuMemPoolGetAttribute` | `CuMemPoolGetAttribute` | memory pools | CUDA 11.2 |
| `cuMemPoolSetAttribute` | `CuMemPoolSetAttribute` | memory pools | CUDA 11.2 |
| `cuMemAllocFromPoolAsync` | `CuMemAllocFromPoolAsync` | memory pools | CUDA 11.2 |

### minimum practical driver version

Only the core set must be present for `Load` to succeed, and those symbols have
been stable across many CUDA releases, so the practical floor for loading is
well below the newest features. The feature groups set their own floors, and
only when used: async allocation needs CUDA 11.2 and graphs need a CUDA 11.x
driver. On an older driver `Load` still succeeds; calling an unavailable feature
returns `ErrSymbolUnavailable` (matchable with `errors.Is`), so the gap is
explicit and local to the call rather than a hard failure at init.

If a required bind fails, `Load` closes the library before returning; a missing
optional symbol is skipped. On successful initialization, the package-global
`cuda` driver keeps the handle alive.

## result mapping

`cudaresult` converts `CUresult` values into Go errors. `Error.Error` renders
known codes as CUDA macro names and unknown codes as `CUDA_ERROR_<number>`.
`Error.Is` compares only the CUDA result code, so operation-specific errors
still match bare sentinels with `errors.Is`.

## executor

CUDA's "current context" is per-OS-thread. Go goroutines move between OS
threads, so a goroutine that called `cuCtxSetCurrent` cannot assume the
context is still current the next time it issues a driver call.

`internal/executor` solves this by owning one goroutine per `Context`,
pinned to a single OS thread with `runtime.LockOSThread`. Every CUDA call
that needs context affinity is submitted to that goroutine and runs there.

```text
caller goroutine -- exec.DoCtx(ctx, fn) --> task channel --> pinned thread
                                                                 ^
                                                                 | runs fn
```

The pinned goroutine never unlocks its OS thread. When `Close` stops the
goroutine, the runtime retires the thread, so there is no thread leak.

`DoCtx` accepts a `context.Context`. Cancellation stops the wait, not the
GPU work; the function still runs to completion on the executor thread and
its result is discarded. The result channel is buffered so the worker does
not block when the caller has walked away.

Synchronous memory copies use a stricter executor path: cancellation can stop
submission, but once a copy is submitted the caller waits until it finishes.
This prevents callers from mutating or reusing Go host slices while CUDA is
still reading or writing them.

Async pinned-memory copies also use the strict submit path, but only wait until
`cuMemcpy*Async` returns. That keeps stream and buffer handles stable while the
driver accepts the work without pretending the GPU copy is complete.

`AllocAsync` and `FreeAsync` (`cuMemAllocAsync` / `cuMemFreeAsync`) use the same
strict `doWait` path. `AllocAsync` must wait for the submit call to return
because that call produces the device pointer the `Buffer` wraps; the allocation
itself is still stream-ordered and is only ready once the stream reaches that
point. `FreeAsync` takes the buffer's write lock and sets the closed flag exactly
like `Close`, so a later `Close` on the same buffer is a no-op rather than a
double free; a failed free leaves the buffer open for retry.

Panics inside `fn` are recovered and surfaced as `*executor.PanicError`;
the executor stays alive so the caller can keep using it or close it.

### allocation-free hot paths

Every call crosses the executor, so its fixed per-call cost matters. The
completion channel is recycled through a `sync.Pool`, and the executor accepts a
`Job` (a `Run() error` method) as well as a plain `func`. The hot copy, memset,
and launch paths submit a pooled `Job` value (`memOp`, `launchOp`,
`graphLaunchOp`) instead of a closure: a pooled pointer passed as an interface
does not allocate, where a per-call closure would. The result is that a steady
copy or `LaunchPacked` loop allocates nothing per call. Cold paths (module load,
context setup, allocation) still use the plain `func` form for clarity, wrapped
internally as a `Job`.

## host pointers in copy paths

`cudasys` declares host-buffer pointers as `*byte`. The `cuda` layer holds a
typed Go slice (`[]T`) and converts to `*byte` at the call site:

```go
srcPtr := (*byte)(unsafe.Pointer(&src[0]))
// ... submit copy task ...
runtime.KeepAlive(src)
```

`runtime.KeepAlive` keeps the slice reachable until after the submitted copy
finishes. Empty slices are rejected at the `cuda` layer before any unsafe
conversion runs.

## pinned host memory

`cuMemAllocHost_v2` returns a host pointer to page-locked memory. The
`HostBuffer[T]` wrapper stores that pointer plus an element count and
exposes `Slice() []T` via `unsafe.Slice` over the pinned region. The
returned slice header points directly at the pinned memory; reads and
writes are zero-copy.

Pinned memory matters because the CUDA driver can DMA between pinned host
memory and the device without staging through a pageable bounce buffer.
It is also recommended for `cuMemcpy*Async` to get predictable overlap
and best throughput; pageable host regions are accepted by the async
APIs in current drivers but the behavior is less predictable. The public async
copy API therefore accepts `HostBuffer` only.

`Buffer.CopyFromHost` and `Buffer.CopyToHost` hold the source/destination
`HostBuffer`'s `sync.RWMutex` read lock across the executor call so
`HostBuffer.Close` cannot race with an in-flight copy. The raw-slice
copy methods (`CopyFrom` / `CopyTo` with `host.Slice()`) do not have this
guarantee because the slice header carries no back-reference to the
`HostBuffer`; the safe path uses the typed methods.

Both `cuMemAllocHost_v2` and `cuMemFreeHost` run on the context executor
via the same strict `doWait` path used by `cuMemAlloc_v2` / `cuMemFree_v2`:
cancellation can stop submission but not abandon an in-flight call.

`Buffer.CopyFromHostAsync` and `Buffer.CopyToHostAsync` hold the stream,
device-buffer, and host-buffer read locks while submitting the async copy.
Those locks protect the enqueue call only. The caller must synchronize the
stream before reading async copy results or closing resources touched by the
queued copy.

Events use the same resource pattern as streams: `Event.Close` takes the event
write lock, while `Record`, `Query`, `Synchronize`, `Elapsed`, and
`Stream.WaitEvent` take read locks. `Record` and `WaitEvent` lock the stream
first and the event second; async-copy submission follows the same idea but
then also locks the device and host buffers it touches. `Elapsed` locks both
events in pointer order so concurrent `a.Elapsed(b)` and `b.Elapsed(a)` cannot
deadlock.

`Event.Record` and `Stream.WaitEvent` use the strict `doWait` executor path
because they enqueue ordering work and need the stream/event handles to remain
valid until the driver accepts that work. `Event.Synchronize` is cancellable
like stream synchronization: cancellation stops the caller's wait, not the GPU
work or the underlying CUDA wait already running on the executor.

## PTX null-termination

`cuModuleLoadData` accepts two distinct kinds of input pointer: a
null-terminated **PTX text** image, or a **cubin / fatbin binary** image
which the driver parses through its own header rather than relying on a
terminator. PTX text produced by `nvcc -ptx` or hand-authored PTX often
omits a trailing zero, so the wrapper makes it safe regardless of source.

`Context.LoadModule` inspects the last byte of the caller's slice: if it
is already `0`, the slice is passed through unchanged; otherwise the
wrapper allocates a fresh `len(image)+1` buffer, copies the bytes, and
lets the trailing zero serve as the terminator. This is harmless for
binary cubin/fatbin images since the driver parses them by header. The
caller's slice is never mutated. `runtime.KeepAlive` keeps the chosen
buffer reachable across the executor call so the GC cannot reclaim it
while the driver is still reading.

`Module.Function` always allocates a `len(name)+1` byte buffer and copies
the Go string into it so the trailing zero is guaranteed. Names
containing an embedded `\x00` are rejected up front with
`ErrInvalidFunctionName`; otherwise CUDA would silently truncate the
name at the first null and bind the wrong kernel. The same `KeepAlive`
discipline applies.

`cuModuleLoadData`, `cuModuleUnload`, and `cuModuleGetFunction` all run
on the context executor via the strict `doWait` path. Module lookups
hold the `Module`'s read lock so `Close` cannot unload the module while
a function lookup is in flight; `Close` takes the write lock to drain
in-flight lookups before issuing `cuModuleUnload`.

## kernel argument packing

`cuLaunchKernel` receives `void** kernelParams`: each element points to the
storage holding one argument value. `internal/argpack.Builder` keeps the common
path inline: up to 16 arguments of eight bytes or less are stored inside the
builder itself, with heap-backed spillover only for unusually large or numerous
arguments. `Function.Launch` keeps that storage alive until `cuLaunchKernel`
returns.

`cuda.Arg(buffer)` stores the device pointer value, not the Go `Buffer`
pointer. It takes the buffer read lock while the driver call is in flight so
`Buffer.Close` cannot race with argument extraction. `cuda.ArgValue(value)`
stores fixed-size scalar values directly. Cross-context buffer arguments are
rejected before submission.

## streams

`Context.NewStream` creates streams with `CU_STREAM_NON_BLOCKING` so work
submitted to them does not implicitly synchronize with the legacy default
stream. The no-option path calls `cuStreamCreate`; `WithStreamPriority` switches
creation to `cuStreamCreateWithPriority`. `Stream.Synchronize` uses the
cancellable wait path; `Stream.Close` uses the strict cleanup path and is
retryable on driver failure.

`Function.Launch` still targets the legacy default stream for the simplest
path. `Function.LaunchOn` takes a stream read lock during `cuLaunchKernel`
submission so `Stream.Close` cannot destroy the handle while the launch call is
in flight. CUDA allows stream destruction with queued work still pending; the
driver releases the stream resources after the work completes.

Returning from `cuLaunchKernel` only means the launch was submitted; GPU
execution may continue afterward. The read locks held by `Launch` / `LaunchOn`
protect only submission. Callers must keep buffers and modules open until
synchronization confirms the kernel is done.
