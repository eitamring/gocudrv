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
- `(*Device).PCIBusID() (string, error)` is the PCI bus identifier
  (`domain:bus:device.function`, for example `0000:01:00.0`).
- `(*Device).UUID() (string, error)` is the device UUID in the canonical
  `GPU-8-4-4-4-12` hex form that nvidia-smi reports.

`Ordinal` returns `-1` for a nil `*Device`. The methods that return errors
return `ErrNilDevice` for a nil `*Device` once the driver is initialized.
`PCIBusID` and `UUID` are backed by best-effort bound symbols, so they return
`ErrSymbolUnavailable` on a driver too old to export them.

## attributes

`DeviceAttribute` is a typed `int32` matching CUDA's device attribute numeric
values. Named attributes currently exposed:

| constant | value |
| --- | --- |
| `DeviceAttributeMaxThreadsPerBlock` | 1 |
| `DeviceAttributeMaxBlockDimX` / `Y` / `Z` | 2, 3, 4 |
| `DeviceAttributeMaxGridDimX` / `Y` / `Z` | 5, 6, 7 |
| `DeviceAttributeMaxSharedMemoryPerBlock` | 8 |
| `DeviceAttributeTotalConstantMemory` | 9 |
| `DeviceAttributeWarpSize` | 10 |
| `DeviceAttributeMaxRegistersPerBlock` | 12 |
| `DeviceAttributeClockRate` | 13 |
| `DeviceAttributeMultiprocessorCount` | 16 |
| `DeviceAttributeIntegrated` | 18 |
| `DeviceAttributeCanMapHostMemory` | 19 |
| `DeviceAttributeComputeMode` | 20 |
| `DeviceAttributeConcurrentKernels` | 31 |
| `DeviceAttributePCIBusID` | 33 |
| `DeviceAttributePCIDeviceID` | 34 |
| `DeviceAttributeTCCDriver` | 35 |
| `DeviceAttributeMemoryClockRate` | 36 |
| `DeviceAttributeGlobalMemoryBusWidth` | 37 |
| `DeviceAttributeL2CacheSize` | 38 |
| `DeviceAttributeMaxThreadsPerMultiprocessor` | 39 |
| `DeviceAttributeAsyncEngineCount` | 40 |
| `DeviceAttributeUnifiedAddressing` | 41 |
| `DeviceAttributePCIDomainID` | 50 |
| `DeviceAttributeComputeCapabilityMajor` | 75 |
| `DeviceAttributeComputeCapabilityMinor` | 76 |
| `DeviceAttributeMaxSharedMemoryPerMultiprocessor` | 81 |
| `DeviceAttributeMaxRegistersPerMultiprocessor` | 82 |
| `DeviceAttributeManagedMemory` | 83 |

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
- `(*Context).MemInfo() (free, total uint64, err error)` returns the free and
  total device memory in bytes. The values reflect the whole device, not just
  this context.
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
- `func AllocAsync[T Supported](ctx *Context, stream *Stream, n int) (*Buffer[T], error)`
  enqueues a stream-ordered allocation of `n` elements on `stream` and returns
  once CUDA accepts the work. The memory must not be accessed until `stream`
  reaches this point (for example after `Stream.Synchronize` or a later op on
  the same stream). Rejects the same inputs as `Alloc`, plus `nil` stream,
  closed stream, and a stream from a different context.
- `(*Buffer[T]).FreeAsync(stream *Stream) error` enqueues a stream-ordered free
  on `stream`. Idempotent after a successful free; failed frees leave the buffer
  open so it can be retried. Returns `ErrContextClosed` if the owning context
  was closed first and `ErrContextMismatch` if `stream` belongs to a different
  context.
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
- `(*Buffer[T]).Zero(ctx context.Context) error` sets every byte of the buffer
  to zero and blocks until the memset completes.
- `(*Buffer[T]).ZeroAsync(ctx context.Context, stream *Stream) error` enqueues
  the clear on `stream` and returns once CUDA accepts the work.
- `(*Buffer[T]).Fill(ctx context.Context, v T) error` sets every element to `v`
  using the device memset whose width matches the element size, so it needs no
  host allocation or copy. Blocks until done. The CUDA driver has no 64-bit
  memset, so Fill returns `ErrUnsupportedFillType` for 8-byte element types
  (`int64`, `uint64`, `float64`).
- `(*Buffer[T]).FillAsync(ctx context.Context, stream *Stream, v T) error`
  enqueues the fill on `stream` and returns once CUDA accepts the work. Same
  8-byte restriction as Fill.
- `(*Buffer[T]).CopyToDevice(ctx context.Context, dst *Buffer[T]) error` copies
  to another device buffer of equal length in the same context. Blocks until
  done.
- `(*Buffer[T]).CopyToDeviceAsync(ctx context.Context, stream *Stream, dst *Buffer[T]) error`
  enqueues a device-to-device copy on `stream`.

### offset copies

These copy a subrange instead of the whole buffer, so a tensor or image
consumer can move a slice without a temporary buffer. The offset and count are
in elements, not bytes, and out-of-range requests are rejected before any CUDA
call: a negative offset or non-positive count returns `ErrInvalidLength`, and a
range that does not fit returns `ErrOutOfRange`.

- `(*Buffer[T]).CopyFromAt(ctx, dstOffset int, src []T) error` copies `len(src)`
  elements from the host slice into the buffer starting at `dstOffset`.
- `(*Buffer[T]).CopyToAt(ctx, dst []T, srcOffset int) error` copies `len(dst)`
  elements from the buffer starting at `srcOffset` into the host slice.
- `(*Buffer[T]).CopyToDeviceAt(ctx, dstOffset int, dst *Buffer[T], srcOffset, n int) error`
  copies `n` elements from `srcOffset` in this buffer to `dstOffset` in `dst`
  (same context).
- `(*Buffer[T]).CopyToDeviceAtAsync(ctx, stream *Stream, dstOffset int, dst *Buffer[T], srcOffset, n int) error`
  enqueues that device-to-device offset copy on `stream`.

The async memset and device-to-device copies (offset and whole-buffer) follow
the same lifetime rule as the async host copies: do not close the buffers or the
stream until `Stream.Synchronize` confirms the work is done.

### views

A `View[T]` is a non-owning window into a region of a `Buffer`. It lets a
higher-level library pass a slice of a buffer without duplicating ownership or
risking a double free. A view has no `Close`; the memory is freed only when the
owning `Buffer` is closed.

- `(*Buffer[T]).View(offset, n int) (*View[T], error)` returns a view of `n`
  elements starting at `offset`. Same validation as the offset copies
  (`ErrInvalidLength` / `ErrOutOfRange`), and the buffer must be open.
- `(*View[T]).View(offset, n int) (*View[T], error)` re-slices into a sub-view of
  the same owner.
- `(*View[T]).Len() int`, `(*View[T]).Bytes() uint64`, and
  `(*View[T]).DevicePtr() cudasys.CUdeviceptr` report the view's extent and
  device pointer (the pointer is a raw snapshot, like `Buffer.DevicePtr`).
- `(*View[T]).CopyFrom(ctx, src []T) error` and `(*View[T]).CopyTo(ctx, dst []T) error`
  copy between the host and the view; the slice length must equal the view
  length.

Views are non-owning by construction: they carry no finalizer and expose no
`Close`. Because copies run through the owner, once the owning `Buffer` is closed
every view operation returns `ErrBufferClosed`. `Buffer.Close` is unchanged. For
a host-side subrange, slice the `HostBuffer.Slice()` result directly; a separate
host view type is not provided.

`AllocAsync` and `FreeAsync` are the stream-ordered counterparts of `Alloc` and
`Close`. Allocation, use, and free are ordered on the same stream, so a buffer
returned by `AllocAsync` is safe to use in later work queued on that stream
without an intervening synchronize. Free the buffer with `FreeAsync` on a stream,
or with `Close` once the stream work that uses it has completed. As with the
other async ops, the memory is only valid after the stream reaches the
allocation point, so do not access it from the host or another stream until
`Stream.Synchronize` confirms the allocation is ready.

Stream-ordered allocation is bound best-effort (it needs a CUDA 11.2 driver), so
on an older driver `AllocAsync` and `FreeAsync` return `ErrSymbolUnavailable`.
Use the synchronous `Alloc` / `Close` path there.

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

### registered host memory

`AllocHost` allocates pinned memory the package owns. When you already hold a
host slice (Go-allocated or external) and want the same pinned-transfer
behavior without reallocating, `RegisterHost` page-locks it in place via
`cuMemHostRegister`.

- `func RegisterHost[T Supported](ctx *Context, mem []T) (*RegisteredHost[T], error)`
  page-locks the backing memory of `mem`. Rejects nil context and an empty
  slice, and returns `ErrSymbolUnavailable` on a driver without the symbol.
- `(*RegisteredHost[T]).Slice()`, `Len()`, `Bytes()` report the registered
  region; pass `Slice()` to `Buffer.CopyFrom` / `CopyTo`.
- `(*RegisteredHost[T]).Close()` unregisters. Idempotent; a failed unregister
  leaves it open to retry.

Unlike `AllocHost`, the caller owns the memory: keep the slice alive and
unchanged until `Close`, and free it only after unregistering. Close the
registration before its `Context`.

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

## pitched memory

`PitchedBuffer[T]` is a 2D device allocation whose rows are padded to a
driver-chosen pitch so each row starts at an aligned address (faster than a
packed allocation for 2D access). `Width` and `Height` are in elements; `Pitch`
is the row stride in bytes, at least `Width*sizeof(T)`.

- `func AllocPitched[T Supported](ctx *Context, width, height int) (*PitchedBuffer[T], error)`
  allocates with `cuMemAllocPitch`. Rejects a nil context, non-positive
  dimensions, and byte overflow; returns `ErrSymbolUnavailable` on a driver
  without the symbol.
- `(*PitchedBuffer[T]).Width()`, `Height()`, `Pitch()`, `DevicePtr()` report the
  geometry and device pointer.
- `(*PitchedBuffer[T]).CopyFrom(ctx, src []T)` and `CopyTo(ctx, dst []T)` move a
  packed host slice of `Width*Height` elements to and from the buffer, adding and
  dropping the row padding (`cuMemcpy2D`). The slice length must equal
  `Width*Height`.
- `(*PitchedBuffer[T]).CopyToDevice(ctx, dst *PitchedBuffer[T])` copies into
  another pitched buffer of equal `Width` and `Height` in the same context;
  their pitches may differ.
- `(*PitchedBuffer[T]).Close()` frees with `cuMemFree`. Idempotent.

The 2D copies use the `CUDA_MEMCPY2D` descriptor, so host rows are treated as
packed (`Pitch == Width*sizeof(T)`) while device rows use the allocation pitch.
This stays a generic CUDA primitive: no image or tensor semantics are implied.

## memory pools

`MemoryPool` is a handle to a device's stream-ordered memory pool, the allocator
behind `AllocAsync`. Reusing the pool lets a high-throughput service tune
caching and allocate from it directly. Memory pools need a CUDA 11.2 driver, so
every entry point here returns `ErrSymbolUnavailable` on an older one.

- `(*Context).DefaultMemPool() (*MemoryPool, error)` returns the device's
  default pool. It is owned by the driver, so `MemoryPool` has no `Close`.
- `(*MemoryPool).ReleaseThreshold()` / `SetReleaseThreshold(bytes)` read and set
  the amount of reserved memory the pool holds before returning it to the OS.
- `(*MemoryPool).ReservedMemCurrent()` and `UsedMemCurrent()` report pool usage
  in bytes.
- `func AllocFromPool[T Supported](pool *MemoryPool, stream *Stream, n int) (*Buffer[T], error)`
  allocates `n` elements from `pool`, ordered on `stream`. Like `AllocAsync`, the
  memory is ready once the stream reaches this point; free it with `FreeAsync` or
  `Close`. The stream must belong to the pool's context.

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
- `(*Stream).Query() error` reports whether the stream is idle without blocking:
  `nil` when all work has finished, `ErrNotReady` while work is still pending. It
  polls one stream without synchronizing the whole context.
- `(*Stream).WaitEvent(event *Event, opts ...WaitOption) error` makes later
  work in this stream wait until `event` completes. No public wait options are
  exposed yet; zero options means CUDA's default wait behavior.
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

### polling and timing

`Stream.Query` (or `Event.Query`) lets a caller poll a single stream and do
other work while the GPU runs, instead of blocking the whole context on
`Synchronize`. Pair it with two timing events to measure the GPU time:

```go
start, _ := ctx.NewEvent()
done, _ := ctx.NewEvent()
defer start.Close()
defer done.Close()

start.Record(stream)
fn.LaunchOn(ctx, stream, cfg, args...)
done.Record(stream)

for stream.Query() == cuda.ErrNotReady {
    // do other host work while the GPU runs
}
elapsed, _ := start.Elapsed(done) // GPU time between the two events
```

## events

`Event` marks a position in a stream. Use events to order work across streams
without synchronizing the whole stream back to the CPU.

```go
ready, err := ctx.NewEvent()
if err != nil {
    log.Fatal(err)
}
defer ready.Close()

if err := ready.Record(copyStream); err != nil {
    log.Fatal(err)
}
if err := computeStream.WaitEvent(ready); err != nil {
    log.Fatal(err)
}
```

- `(*Context).NewEvent(opts ...EventOption) (*Event, error)` creates an event.
- `WithEventBlockingSync()` asks CUDA to block the host thread while waiting on
  this event instead of spin-waiting.
- `WithEventDisableTiming()` disables timestamp recording. Use it for ordering
  events that will not be used with `Elapsed`.
- `(*Event).Record(stream *Stream) error` enqueues the event in `stream`.
- `(*Event).Query() error` returns nil if the event is complete,
  `ErrNotReady` if it is still pending, or another CUDA error if the driver
  reports one.
- `(*Event).Synchronize(ctx context.Context) error` waits until the event
  completes. Canceling `ctx` stops the caller's wait; queued GPU work continues.
- `(*Event).Elapsed(end *Event) (time.Duration, error)` returns GPU time
  between two recorded timing-enabled events.
- `(*Event).Close() error` destroys the event. Idempotent after a successful
  destroy; failed destroys leave the event open so `Close` can be retried.

Nil event methods return `ErrNilEvent`. Methods called after successful close
return `ErrEventClosed`. Events and streams must belong to the same `Context`;
cross-context waits or elapsed-time calls return `ErrContextMismatch`.

Record and wait methods do not take `context.Context` because they enqueue only
a small ordering marker. This is intentionally different from async memory
copies, where cancellation can still matter because the driver may spend real
time accepting the transfer request. `Event.Synchronize` is the wait operation,
so it is the cancellable method.

**Timing rule:** both events passed to `Elapsed` must have timing enabled and
must have completed. If either event is not ready yet, CUDA can return
`ErrNotReady`. If an event was created with `WithEventDisableTiming`, `Elapsed`
returns `ErrEventTimingDisabled` before calling CUDA.

**Lifetime rule:** close events before their owning `Context`. If a queued
stream wait still references an event, `Event.Close` is still safe: CUDA defers
resource cleanup until pending references complete. This is different from
buffers and pinned host memory, where closing too early can free memory the GPU
is still touching.

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
- `(*Context).LoadModuleEx(image []byte, opts JITOptions) (*Module, JITLog, error)`
  loads with JIT options via `cuModuleLoadDataEx` and returns the driver's info
  and error logs. The returned `JITLog.Error` is filled even when the load
  fails, so a PTX compile error surfaces useful diagnostics. `JITOptions` carries
  `LogBufferBytes` (log buffer size, default when `<= 0`) and `MaxRegisters`
  (`CU_JIT_MAX_REGISTERS`, `0` leaves the driver default). The simple
  `LoadModule` is unchanged.
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
- `ArgDevicePtr(ptr)` passes a raw `cudasys.CUdeviceptr` for an advanced caller
  that holds one directly (for example from a sibling module or
  `Buffer.DevicePtr`). It tracks no lifetime and takes no lock, so the caller
  must keep the allocation alive across the launch.
- `ArgRaw(value, size)` passes `size` bytes at `value` for argument types the
  scalars do not cover (a small vector or struct). The bytes are copied when the
  launch is built; `value` must be non-nil and `size` in `(0, 4096]`, else
  `ErrNilKernelArg` / `ErrInvalidArgSize`.
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

Prefer `Arg(buffer)` and `ArgValue` for everything they cover: `Arg` holds the
buffer's read lock across submission so the device pointer cannot be freed
mid-launch, and `ArgValue` is type-checked. `ArgDevicePtr` and `ArgRaw` are
escape hatches with neither guarantee, for raw handles from other CUDA code or
argument types the scalars cannot express; the caller owns correctness.

## occupancy

Occupancy is how many of a multiprocessor's warp slots a kernel can fill. These
helpers let you size a launch by what the hardware can actually run instead of
guessing a block size.

```go
minGrid, blockSize, err := fn.SuggestedBlockSize(0, 0)
cfg, err := fn.SuggestedConfig1D(n, 0) // occupancy-picked block size for n elements
```

- `(*Function).MaxActiveBlocksPerSM(blockSize, dynamicSharedMem int) (int, error)`
  returns the maximum blocks resident per multiprocessor at that block size and
  dynamic shared memory, via `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
  `blockSize` must be positive.
- `(*Function).SuggestedBlockSize(dynamicSharedMem, blockSizeLimit int) (minGridSize, blockSize int, err error)`
  asks `cuOccupancyMaxPotentialBlockSize` for a block size that maximizes
  occupancy and the minimum grid to fill the device. Pass `blockSizeLimit` 0 for
  no cap. The dynamic-shared-memory callback is always null.
- `(*Function).SuggestedConfig1D(n, dynamicSharedMem int) (LaunchConfig, error)`
  folds `SuggestedBlockSize` and `LaunchConfig1D` into one call: a ready 1D config
  covering `n` elements at the occupancy-maximizing block size.

## kernel attributes

Kernels can be inspected and tuned through `cuFuncGetAttribute` /
`cuFuncSetAttribute`. The most common need is raising the dynamic shared memory a
kernel may request: launches default to a 48 KB cap, and a kernel that opts into
more (many attention and GEMM kernels do) must raise it first or the launch
fails.

```go
if err := fn.SetMaxDynamicSharedMemory(96 * 1024); err != nil {
    return err
}
regs, _ := fn.Attribute(cuda.FuncAttrNumRegs)
```

- `(*Function).SetMaxDynamicSharedMemory(bytes int) error` raises the dynamic
  shared-memory limit for launches of the function.
- `(*Function).SetAttribute(attr FunctionAttribute, value int) error` and
  `(*Function).Attribute(attr FunctionAttribute) (int, error)` set and read any
  attribute by its `FuncAttr*` constant.
- All return `ErrSymbolUnavailable` on a driver without the attribute symbols.

## peer access and pointers

For multi-GPU work, a context can be granted direct access to another context's
device memory, and buffers can be copied device-to-device across contexts.

- `(*Device).CanAccessPeer(peer *Device) (bool, error)` reports whether direct
  peer access is possible (`cuDeviceCanAccessPeer`).
- `(*Context).EnablePeerAccess(peer *Context) error` /
  `(*Context).DisablePeerAccess(peer *Context) error` grant and revoke access to
  `peer`'s memory from this context (`cuCtxEnablePeerAccess`).
- `(*Buffer[T]).CopyToPeer(ctx, dst *Buffer[T]) error` copies into an equal-length
  buffer in another context via `cuMemcpyPeer`; it blocks until the copy
  completes. Enabling peer access first is not required but makes it faster.
- `(*Context).PointerMemoryType(ptr) (MemoryType, error)` reports whether a
  pointer addresses host, device, array, or unified memory
  (`cuPointerGetAttribute`).

These all return `ErrSymbolUnavailable` on a driver that lacks the underlying
symbol.

## device globals

`Module.Global` looks up a `__device__` or `__constant__` variable in a loaded
module so the host can read and write it directly.

```go
g, err := mod.Global("g_counter")
cuda.WriteGlobal(ctx, g, []uint32{1, 2, 3, 4})
out := make([]uint32, 4)
cuda.ReadGlobal(ctx, out, g)
```

- `(*Module).Global(name string) (*Global, error)` resolves the symbol via
  `cuModuleGetGlobal` and returns a handle carrying its device pointer and size.
- `(*Global).Bytes() uint64` is the size of the global; `(*Global).Name() string`
  is the symbol name.
- `WriteGlobal[T](ctx, g, vals)` and `ReadGlobal[T](ctx, dst, g)` copy between a
  host slice and the global using `cuMemcpyHtoD` / `cuMemcpyDtoH`. The byte size
  of the slice must be greater than zero and must not exceed `g.Bytes()`, else
  `ErrLengthMismatch`.

A `Global` is tied to its `Module`: once `Module.Close` succeeds the handle is
invalid.

## graphs

A CUDA graph records a sequence of stream work once and replays it with much
lower per-launch overhead than submitting each operation again. Capture work on
a stream, end the capture to get a `Graph`, instantiate it into a `GraphExec`,
then launch that executable repeatedly.

The graph entry points are bound best-effort (they need a CUDA 11.x driver), so
on an older driver the capture, instantiate, and launch calls return
`ErrSymbolUnavailable`.

```go
stream.BeginCapture(cuda.CaptureModeThreadLocal)
fn.LaunchOn(ctx, stream, cfg, args...) // recorded, not run
g, err := stream.EndCapture()
defer g.Close()
exec, err := g.Instantiate()
defer exec.Close()
exec.Launch(ctx, stream) // replay, near-zero launch overhead
stream.Synchronize(ctx)
```

- `(*Stream).BeginCapture(mode CaptureMode) error` puts the stream into capture
  mode via `cuStreamBeginCapture`. `CaptureModeGlobal` is the default;
  `CaptureModeThreadLocal` and `CaptureModeRelaxed` relax the cross-thread safety
  checks.
- `(*Stream).EndCapture() (*Graph, error)` ends capture and returns the recorded
  graph (`cuStreamEndCapture`).
- `(*Graph).Instantiate() (*GraphExec, error)` compiles the graph into an
  executable (`cuGraphInstantiateWithFlags`).
- `(*GraphExec).Launch(ctx, stream) error` enqueues the executable on `stream`
  (`cuGraphLaunch`). It returns after the driver accepts the work, not after the
  GPU finishes; synchronize before reading outputs.
- `(*GraphExec).Update(graph *Graph) error` re-applies a recaptured graph's node
  parameters to the executable without re-instantiating, which is cheap for
  repeated fixed-shape work. It binds the legacy four-argument `cuGraphExecUpdate`
  (CUDA 10.2+) and inspects its update-result out-parameter, so it returns
  `ErrGraphExecUpdateFailure` when the driver declines the update (for example a
  topology change) even if the call itself reports success; re-instantiate then.
  It returns `ErrSymbolUnavailable` on a driver that lacks the symbol. The CUDA 12
  result-info form (`cuGraphExecUpdate_v2`) is intentionally not used.
- `(*Graph).Close()` and `(*GraphExec).Close()` release the handles
  (`cuGraphDestroy`, `cuGraphExecDestroy`).

For repeated work whose shape is fixed but whose parameters change, instantiate
once and update in place each iteration:

```go
exec, _ := g.Instantiate()
defer exec.Close()
for step := 0; step < n; step++ {
    g2, _ := recapture(stream) // capture the same shape with new parameters
    if err := exec.Update(g2); err != nil {
        exec.Close()
        exec, _ = g2.Instantiate() // topology changed; re-instantiate
    }
    g2.Close()
    exec.Launch(ctx, stream)
}
```

**Lifetime rule:** both `Graph` and `GraphExec` are owned by the `Context` and
must be closed before it. The buffers, module, and arguments referenced by the
captured work must stay alive and unchanged for as long as the `GraphExec` may
be launched.

## raw handles (advanced)

These accessors expose the underlying CUDA driver handles so a sibling module
(for example a cuBLAS, cuDNN, or TensorRT binding) can share this package's
context, streams, events, and device buffers instead of creating its own. They
return `cudasys` types directly.

- `(*Context).Raw() cudasys.CUcontext` is the primary context handle.
- `(*Context).Driver() *cudasys.Driver` is the loaded driver with every bound
  entry point, so a sibling can issue driver calls through the library gocudrv
  already opened instead of loading `libcuda` a second time.
- `(*Stream).Raw() cudasys.CUstream` is the stream handle.
- `(*Event).Raw() cudasys.CUevent` is the event handle.
- `(*Buffer[T]).DevicePtr() cudasys.CUdeviceptr` is the device pointer; pair it
  with `Len` and `Bytes` for the element count and byte size.

Each returns the zero value (or nil) on a nil receiver.

**These bypass the safety guarantees of the typed API.** A handle is valid only
while the Go value that owns it is open; using one after `Close` is undefined
behavior that gocudrv cannot detect. CUDA's current context is per OS thread, so
any raw driver call you make yourself must ensure the right context is current on
the calling thread (the typed API does this on its pinned executor). The returned
handle is a snapshot and is not protected by the type's lock. Prefer the typed
methods; reach for these only when handing a raw handle to other CUDA code.

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
ErrSystemDriverMismatch, ErrStreamCaptureUnsupported, ErrStreamCaptureInvalidated,
ErrStreamCaptureMerge, ErrStreamCaptureUnmatched, ErrStreamCaptureUnjoined,
ErrStreamCaptureIsolation, ErrStreamCaptureImplicit, ErrCapturedEvent,
ErrStreamCaptureWrongThread, ErrTimeout, ErrGraphExecUpdateFailure,
ErrExternalDevice, ErrUnknown
```

Go-side sentinels:

- `ErrSymbolUnavailable`: an optional feature symbol (async allocation, occupancy, graphs, or device diagnostics) was not present in the loaded driver, so that call cannot run. Core APIs are unaffected, so this is local to the feature rather than a load-time failure.
- `ErrInvalidOrdinal`: `GetDevice` rejected the ordinal before calling CUDA.
- `ErrNilDevice`: a method was called on a nil `*Device`.
- `ErrNilContext`: a method was called on a nil `*Context`.
- `ErrContextClosed`: a method was called on a `*Context` after `Close`.
- `ErrNilBuffer`: a method was called on a nil `*Buffer[T]` or nil `*HostBuffer[T]`.
- `ErrBufferClosed`: a method was called on a `*Buffer[T]` or `*HostBuffer[T]` after `Close`.
- `ErrLengthMismatch`: a copy was given mismatched or empty slices/buffers.
- `ErrInvalidLength`: `Alloc` or `AllocHost` was given a non-positive or overflowing element count, an offset copy was given a negative offset or non-positive count, or `LoadModuleEx` was given a log buffer size that is negative or larger than the cap.
- `ErrOutOfRange`: an offset copy's range (offset plus count) does not fit the buffer.
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
- `ErrInvalidArgSize`: a raw kernel argument had an unsupported size.
- `ErrContextMismatch`: a kernel argument or stream belongs to a different context from the function.
- `ErrUnsupportedFillType`: `Buffer.Fill` or `FillAsync` was called on an 8-byte element type, which the driver has no memset for.
- `ErrInvalidBlockSize`: an occupancy query was given a block size that is not positive or does not fit CUDA's C `int`.
- `ErrNilEvent`: a method was called on a nil `*Event`.
- `ErrEventClosed`: a method was called on an `*Event` after `Close`.
- `ErrEventTimingDisabled`: `Event.Elapsed` was called on an event created without timing.
- `ErrNilGlobal`: a method was called on a nil `*Global`.
- `ErrEmptyGlobalName`: `Module.Global` was given an empty name.
- `ErrInvalidGlobalName`: `Module.Global` was given a name containing a null byte.
- `ErrNilGraph` / `ErrGraphClosed`: a method was called on a nil or closed `*Graph`.
- `ErrNilGraphExec` / `ErrGraphExecClosed`: a method was called on a nil or closed `*GraphExec`.
- `ErrNilMemPool`: a method was called on a nil `*MemoryPool`.

Returned CUDA errors for codes outside the table still match with:

```go
errors.Is(err, &cuda.Error{Code: code})
```
