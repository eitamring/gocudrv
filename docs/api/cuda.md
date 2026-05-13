# public API

The `cuda` package is the user-facing layer. It owns process-wide driver
initialization and exposes Go-friendly device queries.

## initialization

```go
if err := cuda.Init(); err != nil {
    log.Fatal(err)
}
```

`Init` loads the CUDA driver library and calls `cuInit(0)`. It is idempotent:
after the first successful call, later calls return nil without reloading. If
`cuInit` fails, the library handle is closed so retries do not leak.

`DriverVersion` returns the installed driver version using the CUDA convention.
For example, `12030` means CUDA 12.3.

```go
v, err := cuda.DriverVersion()
fmt.Printf("driver: %d.%d\n", v/1000, (v%1000)/10)
```

## devices

```go
n, err := cuda.DeviceCount()
d, err := cuda.GetDevice(0)
name, err := d.Name()
mem, err := d.TotalMemory()
maj, min, err := d.ComputeCapability()
sms, err := d.Attribute(cuda.DeviceAttributeMultiprocessorCount)
```

`DeviceCount` returns the number of CUDA-capable devices visible to the driver.
`GetDevice` validates the ordinal against `[0, count)` before calling
`cuDeviceGet`.

`Device` is an opaque handle returned by `GetDevice`.

- `(*Device).Ordinal() int`
- `(*Device).Name() (string, error)`
- `(*Device).TotalMemory() (uint64, error)`
- `(*Device).ComputeCapability() (major, minor int, err error)`
- `(*Device).Attribute(attr DeviceAttribute) (int, error)`

`Ordinal` returns `-1` for a nil `*Device`. The methods that return errors
return `ErrNilDevice` for a nil `*Device` once the driver is initialized.

## attributes

`DeviceAttribute` is a typed `int32` matching CUDA's device attribute numeric
values. Named attributes currently exposed:

| constant | value |
| --- | --- |
| `DeviceAttributeMaxThreadsPerBlock` | 1 |
| `DeviceAttributeWarpSize` | 10 |
| `DeviceAttributeClockRate` | 13 |
| `DeviceAttributeMultiprocessorCount` | 16 |
| `DeviceAttributeIntegrated` | 18 |
| `DeviceAttributeConcurrentKernels` | 31 |
| `DeviceAttributeMemoryClockRate` | 36 |
| `DeviceAttributeGlobalMemoryBusWidth` | 37 |
| `DeviceAttributeComputeCapabilityMajor` | 75 |
| `DeviceAttributeComputeCapabilityMinor` | 76 |

Pass `cuda.DeviceAttribute(value)` for CUDA attributes not yet named.

## errors

`cuda.Error` is an alias for `cudaresult.Error`. It carries the raw CUDA result
code and the operation that returned it. Compare errors with `errors.Is`.

```go
if err := cuda.Init(); errors.Is(err, cuda.ErrOperatingSystem) {
    // OS-level call inside cuInit failed.
}
```

CUDA result sentinels include:

```text
ErrInvalidValue, ErrOutOfMemory, ErrNotInitialized, ErrDeinitialized,
ErrProfilerDisabled, ErrStubLibrary, ErrDeviceUnavailable, ErrNoDevice,
ErrInvalidDevice, ErrDeviceNotLicensed, ErrInvalidImage, ErrInvalidContext,
ErrNoBinaryForGPU, ErrInvalidPTX, ErrUnsupportedPTXVersion, ErrInvalidSource,
ErrFileNotFound, ErrSharedObjectSymbolNotFound, ErrSharedObjectInitFailed,
ErrOperatingSystem, ErrInvalidHandle, ErrIllegalState, ErrNotFound,
ErrNotReady, ErrIllegalAddress, ErrLaunchOutOfResources, ErrLaunchTimeout,
ErrLaunchFailed, ErrNotPermitted, ErrNotSupported, ErrSystemNotReady,
ErrSystemDriverMismatch, ErrUnknown
```

Go-side sentinels:

- `ErrInvalidOrdinal`: `GetDevice` rejected the ordinal before calling CUDA.
- `ErrNilDevice`: a method was called on a nil `*Device`.

Returned CUDA errors for codes outside the table still match with:

```go
errors.Is(err, &cuda.Error{Code: code})
```
