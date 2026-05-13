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

Panics inside `fn` are recovered and surfaced as `*executor.PanicError`;
the executor stays alive so the caller can keep using it or close it.
