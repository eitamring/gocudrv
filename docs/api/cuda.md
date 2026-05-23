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
- `(*Context).StreamPriorityRange() (least, greatest int, err error)` returns
  the meaningful CUDA stream-priority interval. Lower numbers mean higher
  priority, so the interval is `[greatest, least]`. Devices without priority
  support return `(0, 0)`.
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
that only wait (`Synchronize` and stream synchronization) take
`context.Context`.

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
- `(*Buffer[T]).CopyFromHost(ctx context.Context, src *HostBuffer[T]) error`
  copies from a pinned `HostBuffer`. Holds the host buffer's read lock for
  the duration of the copy so `HostBuffer.Close` cannot free the pinned
  memory while CUDA is still reading. Prefer this over `CopyFrom` with
  `host.Slice()` when the source is pinned.
- `(*Buffer[T]).CopyToHost(ctx context.Context, dst *HostBuffer[T]) error`
  copies to a pinned `HostBuffer`. Same lock-holding guarantee. Prefer
  over `CopyTo` with `host.Slice()` when the destination is pinned.
- `(*Buffer[T]).CopyFromHostAsync(ctx context.Context, stream *Stream, src *HostBuffer[T]) error`
  enqueues a pinned host-to-device copy on `stream`.
- `(*Buffer[T]).CopyToHostAsync(ctx context.Context, stream *Stream, dst *HostBuffer[T]) error`
  enqueues a device-to-pinned-host copy on `stream`.

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
cancel before the operation is submitted. Synchronous copy cancellation
semantics:

- If `ctx` is already canceled before the call submits to the executor,
  the underlying CUDA copy does not run and the call returns `ctx.Err()`.
- If `ctx` is canceled after submission, the call still waits for the copy to
  finish. This keeps the host slice exclusively owned by the call while CUDA is
  reading or writing it.

Async pinned-copy methods return after CUDA accepts the work, not after the GPU
copy finishes. If `ctx` is already canceled before submission, the copy is not
enqueued and the call returns `ctx.Err()`. If cancellation happens after
submission, the call still waits until the enqueue call returns so the stream
and buffer handles remain valid during submission.

An error returned after submission may come from the driver while accepting the
work. Treat the stream as needing normal error handling; a later
`Stream.Synchronize` may also report CUDA work failure.

**Async lifetime rule:** after `CopyFromHostAsync`, do not mutate the source
`HostBuffer` and do not close the source, destination, or stream until
`Stream.Synchronize` confirms the copy is done. After `CopyToHostAsync`, do not
read the destination `HostBuffer` and do not close the source, destination, or
stream until synchronization completes.

**Lifetime rule:** a `Buffer` must be closed before its owning `Context`
is closed. After the `Context` is closed, `Buffer.Close` cannot reach the
executor and returns `ErrContextClosed`; CUDA reclaims the device memory
when the primary-context retain count drops to zero, but the wrapper
cannot guarantee that ordering. Pair every `Alloc` with `defer buf.Close()`
and close every buffer before the context.

## pinned host memory

`HostBuffer[T]` is a typed handle to a region of page-locked (pinned)
host memory owned by a `Context`. CUDA can DMA directly to and from this
memory, skipping its internal staging buffer, so transfers are faster
than copies from pageable Go slices. Pinned memory is also recommended
for predictable async-copy overlap and best throughput; pageable memory is
supported by CUDA but tends to be slower and less predictable.

```go
host, err := cuda.AllocHost[float32](ctx, 1024)
if err != nil {
    log.Fatal(err)
}
defer host.Close()

s := host.Slice()
for i := range s {
    s[i] = float32(i)
}

// Prefer the *Host methods when copying to/from a HostBuffer. They hold
// the host buffer's read lock for the duration of the copy, so it cannot
// be closed (and the pinned memory cannot be freed) while CUDA reads it.
if err := buf.CopyFromHost(context.Background(), host); err != nil {
    log.Fatal(err)
}
```

- `func AllocHost[T Supported](ctx *Context, n int) (*HostBuffer[T], error)`
  allocates `n` elements of pinned host memory. Rejects nil context, closed
  context, `n <= 0`, and byte-size overflow.
- `(*HostBuffer[T]).Len() int` returns the element count.
- `(*HostBuffer[T]).Bytes() uint64` returns the total byte size.
- `(*HostBuffer[T]).Slice() []T` returns a `[]T` view backed by the pinned
  memory. The slice can be read and written directly. Returns `nil` if the
  buffer is nil or has been closed.
- `(*HostBuffer[T]).Close() error` releases the pinned memory. Idempotent
  after a successful free; failed frees leave the buffer open so `Close`
  can be retried.

The slice returned by `Slice` becomes invalid after `Close`. Do not retain
it past that point; using it after `Close` reads or writes freed memory.

Use `Buffer.CopyFromHost` / `CopyToHost` to move data between a `Buffer`
and a `HostBuffer`. They lock the host buffer against concurrent `Close`
for the duration of the copy. `Buffer.CopyFrom` / `CopyTo` with
`host.Slice()` still work for CPU-only access patterns, but they cannot
prevent another goroutine from closing the `HostBuffer` mid-copy, so the
typed methods are the safe path for CUDA transfers.

Use `Buffer.CopyFromHostAsync` / `CopyToHostAsync` with an explicit `Stream`
when you want to enqueue copies that can overlap with other stream work. These
methods are pinned-buffer only. There is intentionally no
`CopyFromAsync(ctx, stream, []T)` API: after an async enqueue returns, the GPU
may still read or write the host memory, and a normal Go slice has no CUDA
lifetime handle for this package to protect. Do not work around this with
`unsafe.Pointer(&slice[0])`; use `AllocHost` for async transfers.

Pinned memory is an optional faster path, not a replacement. Pageable Go
slices are still accepted by `Buffer.CopyFrom` / `CopyTo`. Use pinned
memory for repeated large transfers and for async copies; for tiny
one-off copies the pageable path is fine.

Lifetime rule mirrors `Buffer`: a `HostBuffer` must be closed before its
owning `Context` is closed.

## streams

`Stream` is an ordered queue of GPU work owned by a `Context`. New streams are
created as non-blocking streams, so work submitted to them does not implicitly
synchronize with the legacy default stream. Explicit streams give the API a
place to route independent work and async pinned-memory copies.

```go
stream, err := ctx.NewStream()
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

if err := stream.Synchronize(context.Background()); err != nil {
    log.Fatal(err)
}
```

- `(*Context).NewStream(opts ...StreamOption) (*Stream, error)` creates a
  non-blocking stream.
- `WithStreamPriority(priority int)` requests a CUDA stream priority. Lower
  numbers mean higher priority, and `0` is the default. Use
  `Context.StreamPriorityRange` to discover the meaningful interval for the
  current device; CUDA clamps values outside that range.
- `(*Stream).Synchronize(ctx context.Context) error` waits until preceding
  work in that stream finishes. Canceling `ctx` stops the wait; queued GPU work
  continues.
- `(*Stream).Close() error` destroys the stream. Idempotent after a successful
  destroy; failed destroys leave the stream open so `Close` can be retried.

Nil stream methods return `ErrNilStream`. Methods called after successful close
return `ErrStreamClosed`.

**Lifetime rule:** close streams before their owning `Context`. Destroying a
stream does not wait for already queued GPU work to finish. If you call
`Stream.Close` and then close a buffer or module that queued work still uses,
the GPU may keep touching a resource you just freed. Call `Stream.Synchronize`
before reading outputs or closing anything touched by work submitted to that
stream.

Canceling `Stream.Synchronize` only stops the caller's wait. It does not stop
the queued GPU work or the underlying CUDA synchronization already running on
the executor thread; a later `Stream.Close` will still wait behind that work.

## modules

`Module` is a handle to a loaded PTX or cubin image owned by a `Context`.
Use it to look up kernel functions by name.

```go
ptx, err := os.ReadFile("vector_add.ptx")
if err != nil {
    log.Fatal(err)
}

mod, err := ctx.LoadModule(ptx)
if err != nil {
    log.Fatal(err)
}
defer mod.Close()

fn, err := mod.Function("vector_add")
if err != nil {
    log.Fatal(err)
}
```

- `(*Context).LoadModule(image []byte) (*Module, error)` calls
  `cuModuleLoadData` with the image. PTX images must be null-terminated;
  if the slice is not already, a fresh copy with a trailing null byte is
  passed to the driver so the caller's slice is not mutated.
- `(*Context).LoadModuleFromFile(path string) (*Module, error)` reads the
  file at `path` and forwards the bytes to `LoadModule`. Empty path is
  rejected with `ErrEmptyImage`; read errors are wrapped with the path.
- `(*Module).Function(name string) (*Function, error)` looks up a kernel.
  The name is converted to a null-terminated byte sequence before being
  passed to `cuModuleGetFunction`.
- `(*Module).Close() error` unloads the module. Idempotent after a
  successful unload; failures leave the module open so `Close` can be
  retried.
- `(*Function).Name() string` returns the kernel name used to look up the
  function. Returns `""` for a nil `*Function`.

`LoadModule` and `Module.Close` do not take `context.Context` for the same
reason as the other ownership-managing entry points: partial completion
would leak module state.

**Lifetime rule:** a `Module` must be closed before its owning `Context`
is closed. After the `Context` is closed, `Module.Close` cannot reach the
executor and returns `ErrContextClosed`. Pair every `LoadModule` with
`defer mod.Close()` and close every module before the context. A
`Function` is tied to its `Module`: once `Module.Close` succeeds the
handle is invalid.

## kernel launch

`Function.Launch` enqueues a kernel on the context's legacy default stream.
`Function.LaunchOn` enqueues on an explicit stream. The first release supports
device-buffer pointers and fixed-size scalar values:

```go
cfg := cuda.LaunchConfig1D(n, 256)
if err := fn.Launch(context.Background(), cfg,
    cuda.Arg(a),
    cuda.Arg(b),
    cuda.Arg(out),
    cuda.ArgValue(int32(n)),
); err != nil {
    log.Fatal(err)
}
if err := ctx.Synchronize(context.Background()); err != nil {
    log.Fatal(err)
}
```

```go
stream, err := ctx.NewStream()
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

if err := fn.LaunchOn(context.Background(), stream, cfg,
    cuda.Arg(a),
    cuda.Arg(b),
    cuda.Arg(out),
    cuda.ArgValue(int32(n)),
); err != nil {
    log.Fatal(err)
}
if err := stream.Synchronize(context.Background()); err != nil {
    log.Fatal(err)
}
```

- `LaunchConfig` carries grid, block, and dynamic shared-memory dimensions.
- `LaunchConfig1D(n, blockSize)` builds a one-dimensional config covering
  `n` elements, rounding the grid up.
- `Arg(buffer)` passes a device-buffer pointer.
- `ArgValue(value)` passes a fixed-size scalar value.
- `(*Function).Launch(ctx, cfg, args...)` submits the launch on the legacy
  default stream. Invalid zero dimensions return `ErrInvalidLaunchConfig`.
- `(*Function).LaunchOn(ctx, stream, cfg, args...)` submits on `stream`.
  Nil, closed, or cross-context streams are rejected before submission.

Cancellation can stop submission, but once submitted either launch method waits
until `cuLaunchKernel` returns so temporary Go argument storage remains valid.

**Lifetime rule:** launches are asynchronous with respect to GPU execution.
After either method returns, the kernel may still be running. The launch-time
locks only protect submission, not the whole kernel lifetime. Call
`Context.Synchronize` or `Stream.Synchronize` before reading outputs or closing
any buffer or module the kernel touched.

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
- `ErrNilBuffer`: a method was called on a nil `*Buffer[T]` or nil `*HostBuffer[T]`.
- `ErrBufferClosed`: a method was called on a `*Buffer[T]` or `*HostBuffer[T]` after `Close`.
- `ErrLengthMismatch`: a copy was given mismatched or empty slices/buffers.
- `ErrInvalidLength`: `Alloc` or `AllocHost` was given a non-positive or overflowing element count.
- `ErrNilModule`: a method was called on a nil `*Module`.
- `ErrModuleClosed`: a method was called on a `*Module` after `Close`.
- `ErrEmptyImage`: `LoadModule` was given a nil or empty image, or `LoadModuleFromFile` was given an empty path.
- `ErrEmptyFunctionName`: `Module.Function` was given an empty name.
- `ErrInvalidFunctionName`: `Module.Function` was given a name containing a null byte (CUDA would silently truncate it).
- `ErrNilFunction`: a method was called on a nil `*Function`.
- `ErrNilStream`: a method was called on a nil `*Stream`.
- `ErrStreamClosed`: a method was called on a `*Stream` after `Close`.
- `ErrInvalidStreamPriority`: `WithStreamPriority` received a value that cannot fit in CUDA's C `int` priority parameter.
- `ErrInvalidLaunchConfig`: `Function.Launch` or `LaunchOn` was given zero grid or block dimensions.
- `ErrNilKernelArg`: `Function.Launch` or `LaunchOn` was given a nil `KernelArg`.
- `ErrContextMismatch`: a kernel argument or stream belongs to a different context from the function.

Returned CUDA errors for codes outside the table still match with:

```go
errors.Is(err, &cuda.Error{Code: code})
```
