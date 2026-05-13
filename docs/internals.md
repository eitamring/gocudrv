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
    CuInit               func(flags uint32) CUresult
    CuDriverGetVersion   func(version *int32) CUresult
    CuDeviceGetCount     func(count *int32) CUresult
    CuDeviceGet          func(device *CUdevice, ordinal int32) CUresult
    CuDeviceGetName      func(name *byte, length int32, dev CUdevice) CUresult
    CuDeviceTotalMem     func(bytes *uint64, dev CUdevice) CUresult
    CuDeviceGetAttribute func(value *int32, attr int32, dev CUdevice) CUresult
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

If any bind fails, `Load` closes the library before returning. On successful
initialization, the package-global `cuda` driver keeps the handle alive.

## result mapping

`cudaresult` converts `CUresult` values into Go errors. `Error.Error` renders
known codes as CUDA macro names and unknown codes as `CUDA_ERROR_<number>`.
`Error.Is` compares only the CUDA result code, so operation-specific errors
still match bare sentinels with `errors.Is`.
