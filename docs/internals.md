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
}
```

`cudasys.Load` binds:

- `cuInit`
- `cuDriverGetVersion`
- `cuDeviceGetCount`
- `cuDeviceGet`
- `cuDeviceGetName`
- `cuDeviceTotalMem_v2`
- `cuDeviceGetAttribute`
- `cuCtxGetCurrent`
- `cuCtxSetCurrent`
- `cuCtxSynchronize`
- `cuDevicePrimaryCtxRetain`
- `cuDevicePrimaryCtxRelease_v2`
- `cuMemAlloc_v2`
- `cuMemFree_v2`
- `cuMemcpyHtoD_v2`
- `cuMemcpyDtoH_v2`
- `cuMemAllocHost_v2`
- `cuMemFreeHost`

If any bind fails, `Load` closes the library before returning. On successful
initialization, the package-global `cuda` driver keeps the handle alive.

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

Memory copies use a stricter executor path: cancellation can stop submission,
but once a copy is submitted the caller waits until it finishes. This prevents
callers from mutating or reusing Go host slices while CUDA is still reading or
writing them.

Panics inside `fn` are recovered and surfaced as `*executor.PanicError`;
the executor stays alive so the caller can keep using it or close it.

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
APIs in current drivers but the behavior is less predictable. The async
path lands in the streams PR.

`Buffer.CopyFromHost` and `Buffer.CopyToHost` hold the source/destination
`HostBuffer`'s `sync.RWMutex` read lock across the executor call so
`HostBuffer.Close` cannot race with an in-flight copy. The raw-slice
copy methods (`CopyFrom` / `CopyTo` with `host.Slice()`) do not have this
guarantee because the slice header carries no back-reference to the
`HostBuffer`; the safe path uses the typed methods.

Both `cuMemAllocHost_v2` and `cuMemFreeHost` run on the context executor
via the same strict `doWait` path used by `cuMemAlloc_v2` / `cuMemFree_v2`:
cancellation can stop submission but not abandon an in-flight call.
