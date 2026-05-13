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

## contexts

A `Context` wraps the device's primary context and a pinned-thread executor.
Every driver call that needs context affinity routes through that thread so
"current context" stays stable across goroutines.

```go
dev, _ := cuda.GetDevice(0)
ctx, err := dev.Primary()
if err != nil {
    log.Fatal(err)
}
defer ctx.Close()

if err := ctx.Synchronize(context.Background()); err != nil {
    log.Fatal(err)
}
```

- `(*Device).Primary() (*Context, error)` retains the primary context and
  starts the executor. Rolls back retain and stops the executor on failure.
- `(*Context).Device() *Device` returns the device this context was created
  on.
- `(*Context).Synchronize(ctx context.Context) error` blocks until all
  preceding GPU work finishes or `ctx` is canceled. Canceling stops the
  wait; the GPU work continues regardless.
- `(*Context).Close() error` releases the primary-context retain and stops
  the executor. Idempotent; subsequent calls return the first call's error.
  Methods called after `Close` return `ErrContextClosed`.

Nil `*Context` methods return `ErrNilContext` when they return an error, and
`Device` returns nil.

`Primary` and `Close` do not take a `context.Context`: they mutate
ownership state and partial completion would leak retain counts. Methods
that only wait (`Synchronize`, and future memory and stream operations)
take `context.Context`.

## memory

`Buffer[T]` is a typed handle to a region of device memory owned by a
`Context`. `T` must satisfy `Supported`, which is restricted to fixed-size
numeric scalars (`int8/16/32/64`, `uint8/16/32/64`, `float32`, `float64`).
Structs and unsized `int`/`uint` are intentionally excluded to avoid
alignment hazards.

```go
buf, err := cuda.Alloc[float32](ctx, 1024)
if err != nil {
    log.Fatal(err)
}
defer buf.Close()

src := make([]float32, 1024)
for i := range src {
    src[i] = float32(i)
}

bg := context.Background()
if err := buf.CopyFrom(bg, src); err != nil {
    log.Fatal(err)
}

dst := make([]float32, 1024)
if err := buf.CopyTo(bg, dst); err != nil {
    log.Fatal(err)
}
```

- `func Alloc[T Supported](ctx *Context, n int) (*Buffer[T], error)`
  allocates `n` elements. Rejects `nil` context, closed context, `n <= 0`,
  and byte-size overflow.
- `(*Buffer[T]).Len() int` returns the element count.
- `(*Buffer[T]).Bytes() uint64` returns the total byte size.
- `(*Buffer[T]).Close() error` frees the device memory. Idempotent after a
  successful free; failed frees leave the buffer open so `Close` can be
  retried.
- `(*Buffer[T]).CopyFrom(ctx context.Context, src []T) error` copies host
  to device. Lengths must match.
- `(*Buffer[T]).CopyTo(ctx context.Context, dst []T) error` copies device
  to host. Same shape.

Two free-function wrappers exist for callers who prefer the CUDA-style
naming:

```go
func CopyHtoD[T Supported](ctx context.Context, dst *Buffer[T], src []T) error
func CopyDtoH[T Supported](ctx context.Context, dst []T, src *Buffer[T]) error
```

Both delegate to the methods. Prefer the method form in new code.

`Alloc` and `Buffer.Close` do not take `context.Context` for the same
reason as `Primary` and `Context.Close`: they manage ownership and partial
completion would leak. The copy methods take `context.Context`, but only to
cancel before the operation is submitted. Cancellation semantics:

- If `ctx` is already canceled before the call submits to the executor,
  the underlying CUDA copy does not run and the call returns `ctx.Err()`.
- If `ctx` is canceled after submission, the call still waits for the copy to
  finish. This keeps the host slice exclusively owned by the call while CUDA is
  reading or writing it.

**Lifetime rule:** a `Buffer` must be closed before its owning `Context`
is closed. After the `Context` is closed, `Buffer.Close` cannot reach the
executor and returns `ErrContextClosed`; CUDA reclaims the device memory
when the primary-context retain count drops to zero, but the wrapper
cannot guarantee that ordering. Pair every `Alloc` with `defer buf.Close()`
and close every buffer before the context.

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
- `ErrNilContext`: a method was called on a nil `*Context`.
- `ErrContextClosed`: a method was called on a `*Context` after `Close`.
- `ErrNilBuffer`: a method was called on a nil `*Buffer[T]`.
- `ErrBufferClosed`: a method was called on a `*Buffer[T]` after `Close`.
- `ErrLengthMismatch`: a copy was given mismatched or empty slices.
- `ErrInvalidLength`: `Alloc` was given a non-positive or overflowing element count.

Returned CUDA errors for codes outside the table still match with:

```go
errors.Is(err, &cuda.Error{Code: code})
```
