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
| `DeviceAttributeConcurrentManagedAccess` | 89 |
| `DeviceAttributeCooperativeLaunch` | 95 |

Pass `cuda.DeviceAttribute(value)` for CUDA attributes not yet named.

## contexts

A `Context` wraps the device's primary context and a pinned command executor.
Synchronous copies use a separate executor created on first use, and
synchronization calls run on a small pool of wait lanes created on demand, so
unrelated waits do not queue behind each other. A long copy or GPU wait
therefore does not stop unrelated queries, launches, or async submissions.
Every executor binds the same context on its pinned thread.

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
  starts the command executor. Rolls back retain and stops the executor on
  failure. The copy executor and the wait lanes start lazily.
- `(*Context).Device() *Device` returns the device this context was created
  on.
- `(*Context).StreamPriorityRange() (least, greatest int, err error)` returns
  the meaningful CUDA stream-priority interval. Lower numbers mean higher
  priority, so the interval is `[greatest, least]`. Devices without priority
  support return `(0, 0)`.
- `(*Context).Synchronize(ctx context.Context) error` blocks until all
  preceding GPU work finishes or `ctx` is canceled. Canceling stops the
  caller's wait; the GPU work and underlying driver wait continue regardless.
- `(*Context).MemInfo() (free, total uint64, err error)` returns the free and
  total device memory in bytes. The values reflect the whole device, not just
  this context.
- `(*Context).Close() error` drains all started executors and releases the
  primary-context retain. It is idempotent after success. A failed release
  leaves the context open and retryable. Methods called after a successful
  `Close` return `ErrContextClosed`.

Nil `*Context` methods return `ErrNilContext` when they return an error, and
`Device` returns nil.

`Primary` and `Close` do not take a `context.Context`: they mutate
ownership state and partial completion would leak retain counts. Methods
that only wait (`Synchronize` and stream synchronization) take
`context.Context`. Waits run on a small pool of wait lanes, so unrelated waits
do not queue behind each other. Canceling a wait does not occupy the command
executor, but resource cleanup still waits for the accepted driver wait to
finish.

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
- `(*Buffer[T]).CopyFromHost(ctx context.Context, src PinnedHost[T]) error`
  copies from an allocated or registered pinned host region. It holds the
  host region's read lock through the driver call so it cannot be closed while
  CUDA is reading.
- `(*Buffer[T]).CopyToHost(ctx context.Context, dst PinnedHost[T]) error`
  copies to an allocated or registered pinned host region with the same lock
  guarantee.
- `(*Buffer[T]).CopyFromHostAsync(ctx context.Context, stream *Stream, src PinnedHost[T]) error`
  enqueues a pinned host-to-device copy on `stream`.
- `(*Buffer[T]).CopyToHostAsync(ctx context.Context, stream *Stream, dst PinnedHost[T]) error`
  enqueues a device-to-pinned-host copy on `stream`.

Compatibility note: these four methods previously accepted `*HostBuffer[T]`.
Ordinary calls with a host buffer still compile, but the exact method types
changed. Stored method values and interfaces using the old signatures must use
`PinnedHost[T]` instead.

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

**Async lifetime rule:** after `CopyFromHostAsync`, do not mutate or close the
source `PinnedHost`, destination, or stream until `Stream.Synchronize` confirms
the copy is done. After `CopyToHostAsync`, do not read or close the destination
`PinnedHost`, source, or stream until synchronization completes.

**Lifetime rule:** a `Buffer` must be closed before its owning `Context`
is closed. After the `Context` is closed, `Buffer.Close` cannot reach the
executor and returns `ErrContextClosed`; CUDA reclaims the device memory
when the primary-context retain count drops to zero, but the wrapper
cannot guarantee that ordering. Pair every `Alloc` with `defer buf.Close()`
and close every buffer before the context.

## pinned host memory

`PinnedHost[T]` is the sealed interface shared by `HostBuffer[T]` and
`RegisteredHost[T]`. It exposes `Slice`, `Len`, and `Bytes`, and lets the typed
copy APIs accept either kind while retaining the correct context, lock, and
lifetime checks. Callers outside the package cannot implement it with pageable
memory.

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

Use `Buffer.CopyFromHost` / `CopyToHost` to move data between a `Buffer` and
either `PinnedHost` implementation. They lock the host region against
concurrent `Close` for the duration of the driver call. `Buffer.CopyFrom` /
`CopyTo` with `host.Slice()` still work for CPU-only access patterns, but they
cannot prevent another goroutine from closing the host region mid-copy.

### registered host memory

`AllocHost` allocates pinned memory the package owns. When you already hold a
host slice (Go-allocated or external) and want the same pinned-transfer
behavior without reallocating, `RegisterHost` page-locks it in place via
`cuMemHostRegister`.

- `func RegisterHost[T Supported](ctx *Context, mem []T) (*RegisteredHost[T], error)`
  page-locks the backing memory of `mem`. Rejects nil context and an empty
  slice, and returns `ErrSymbolUnavailable` on a driver without the symbol.
- `(*RegisteredHost[T]).Slice()`, `Len()`, `Bytes()` report the registered
  region. Pass the registration directly to the `*Host` copy methods.
- `(*RegisteredHost[T]).Close()` unregisters. Idempotent; a failed unregister
  leaves it open to retry.

Unlike `AllocHost`, the caller owns the memory: keep the slice alive and
unchanged until `Close`, and free it only after unregistering. Close the
registration before its `Context`.

Use the async `*Host` methods with an explicit `Stream` when you want copies
that can overlap with other stream work. They accept either allocated or
registered pinned memory. There is intentionally no
`CopyFromAsync(ctx, stream, []T)` API: after an async enqueue returns, the GPU
may still read or write the host memory, and a normal Go slice has no CUDA
lifetime handle for this package to protect. Do not work around this with
`unsafe.Pointer(&slice[0])`; use `AllocHost` or `RegisterHost`.

Pinned memory is an optional faster path, not a replacement. Pageable Go
slices are still accepted by `Buffer.CopyFrom` / `CopyTo`. Use pinned
memory for repeated large transfers and for async copies; for tiny
one-off copies the pageable path is fine.

Lifetime rules mirror `Buffer`: close `HostBuffer` and `RegisteredHost` values
before their owning `Context`. Async flat and shaped copies hold locks through
driver submission only. Keep every participating host region, device resource,
and stream open, and do not read or mutate host memory until synchronization.

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
- `CopyFromHostAsync(ctx, stream, src PinnedHost[T])` and
  `CopyToHostAsync(ctx, stream, dst PinnedHost[T])` enqueue packed host copies.
  `CopyToDeviceAsync(ctx, stream, dst)` enqueues the matching pitched
  device-to-device copy. All shapes must match.
- `(*PitchedBuffer[T]).Close()` frees with `cuMemFree`. Idempotent.

The 2D copies use the `CUDA_MEMCPY2D` descriptor, so host rows are treated as
packed (`Pitch == Width*sizeof(T)`) while device rows use the allocation pitch.
This stays a generic CUDA primitive: no image or tensor semantics are implied.

## volumes (3D)

`Volume[T]` is the 3D analogue of `PitchedBuffer`: a `Width`-by-`Height`-by-`Depth`
region (in elements) laid out as `Depth` slices of `Height` padded rows and
copied with `cuMemcpy3D`. It is backed by `cuMemAllocPitch` over `Height*Depth`
rows, so `Pitch` is the driver-chosen row stride shared by every slice.

- `func AllocVolume[T Supported](ctx *Context, width, height, depth int) (*Volume[T], error)`
  allocates the padded region. Rejects a nil context, non-positive dimensions,
  and byte or element-count overflow; returns `ErrSymbolUnavailable` on a driver
  without `cuMemAllocPitch`.
- `(*Volume[T]).Width()`, `Height()`, `Depth()`, `Pitch()`, `DevicePtr()` report
  the geometry and device pointer.
- `(*Volume[T]).CopyFrom(ctx, src []T)` and `CopyTo(ctx, dst []T)` move a packed
  host slice of `Width*Height*Depth` elements to and from the volume, adding and
  dropping the row padding (`cuMemcpy3D`). The slice length must match.
- `CopyFromHostAsync(ctx, stream, src PinnedHost[T])` and
  `CopyToHostAsync(ctx, stream, dst PinnedHost[T])` enqueue the same packed 3D
  geometry with pinned memory.
- `(*Volume[T]).Close()` frees with `cuMemFree`. Idempotent.

The 3D copies use the `CUDA_MEMCPY3D` descriptor with the host side packed and
the device side pitched. Like the 2D case this is a generic memory primitive;
CUDA arrays, textures, and sub-region boxes are out of scope here.

## CUDA arrays and textures

`Array2D[T]` is a CUDA array: a 2D device allocation in an opaque,
texture-optimized layout with no device pointer. A `Texture` is a sampling view
over it that a kernel fetches through the texture cache, with addressing,
filtering, and optional coordinate normalization.

```go
arr, err := cuda.AllocArray2D[float32](ctx, w, h)
err = arr.CopyFrom(context.Background(), pixels)
tex, err := cuda.NewTexture(arr, cuda.TextureConfig{
    AddressMode: cuda.AddressClamp,
    FilterMode:  cuda.FilterLinear,
})
err = fn.Launch(context.Background(), cfg, cuda.ArgTexture(tex), cuda.ArgValue(int32(w)))
```

- `func AllocArray2D[T Supported](ctx *Context, width, height int, opts ...ArrayOption) (*Array2D[T], error)`
  creates the array with `cuArrayCreate` (format derived from `T`, one channel),
  or with `cuArray3DCreate` and the surface load/store flag when
  `WithSurfaceStore()` is passed. CUDA arrays support 1-, 2-, and 4-byte element
  types, so 8-byte types return `ErrUnsupportedElement`.
- `(*Array2D[T]).CopyFrom(ctx, src []T)` and `CopyTo(ctx, dst []T)` move a
  packed host slice of `Width*Height` elements to and from the array
  (`cuMemcpy2D` with an array endpoint). `Width()`, `Height()`, and `Raw()`
  report the geometry and the raw `CUarray` handle.
- `CopyFromHostAsync(ctx, stream, src PinnedHost[T])` and
  `CopyToHostAsync(ctx, stream, dst PinnedHost[T])` enqueue the same array
  transfers with allocated or registered pinned memory.
- `func NewTexture[T Supported](arr *Array2D[T], cfg TextureConfig) (*Texture, error)`
  creates a texture object over the array (`cuTexObjectCreate`).
  `TextureConfig` sets the `AddressMode` (`AddressWrap`/`Clamp`/`Mirror`/`Border`),
  the `FilterMode` (`FilterPoint`/`FilterLinear`; linear requires a float
  element type), and `NormalizedCoordinates`. Integer element types are read as
  integers automatically. Wrap and mirror addressing take effect only with
  normalized coordinates.
- `ArgTexture(t *Texture)` passes the texture to a kernel (the parameter is a
  `cudaTextureObject_t`); like `Arg` it holds the texture's read lock across
  submission. `(*Texture).Raw()` exposes the `CUtexObject` handle for sibling
  libraries.
- `(*Texture).Close()` then `(*Array2D[T]).Close()`: close textures before the
  array they sample, and the array before the context. Both are idempotent and
  leave the handle open to retry on a failed destroy.

### surfaces

A `Surface` is the writable counterpart of a `Texture`: a kernel reads and
writes the array through it with `surf2Dread`/`surf2Dwrite` (no filtering or
normalized coordinates; exact element access).

```go
arr, err := cuda.AllocArray2D[uint32](ctx, w, h, cuda.WithSurfaceStore())
surf, err := cuda.NewSurface(arr)
err = fn.Launch(context.Background(), cfg, cuda.ArgSurface(surf), cuda.ArgValue(int32(w)))
```

- `AllocArray2D` takes options: `WithSurfaceStore()` allocates through
  `cuArray3DCreate` with the surface load/store flag, which surfaces require.
  Plain arrays keep the old path and cannot back a surface.
- `func NewSurface[T Supported](arr *Array2D[T]) (*Surface, error)` creates the
  surface object (`cuSurfObjectCreate`). An array allocated without
  `WithSurfaceStore` is rejected with `ErrNoSurfaceStore`.
- `ArgSurface(s *Surface)` passes it to a kernel (the parameter is a
  `cudaSurfaceObject_t`); `(*Surface).Raw()` exposes the `CUsurfObject`.
- `(*Surface).Close()` destroys it; close surfaces (and textures) before the
  array, and the array before the context.

Layered/3D arrays are not covered yet. Together textures and surfaces complete
the 2D array story: stage data in, sample reads through the texture cache,
write results back through a surface.

## interprocess sharing (IPC)

Device memory and events can cross process boundaries on one machine: a
producer exports an opaque 64-byte handle, ships it over any channel (pipe,
socket, file), and the consumer maps the same allocation.

```go
h, err := buf.IPCHandle()            // in the exporting process
raw := h.Bytes()                     // send these 64 bytes to the other process

imp, err := cuda.OpenIPCBuffer[float32](ctx, cuda.IPCMemHandleFromBytes(raw), n)
err = imp.CopyTo(context.Background(), out)   // reads the exporter's memory
err = imp.Close()                             // unmaps here; never frees theirs
```

- `(*Buffer[T]).IPCHandle() (IPCMemHandle, error)` exports the allocation
  (`cuIpcGetMemHandle`). Keep the buffer open while any process has it mapped.
- `IPCMemHandle.Bytes()` / `IPCMemHandleFromBytes(b)` convert the handle to and
  from its raw 64 bytes for transport.
- `func OpenIPCBuffer[T Supported](ctx *Context, h IPCMemHandle, n int) (*IPCBuffer[T], error)`
  maps the exporter's allocation (`cuIpcOpenMemHandle`, lazy peer access). The
  handle does not carry the size, so `n` must match what the exporter
  allocated. Opening a handle in the process that exported it fails with
  `ErrInvalidContext`; only plain `Alloc` memory is exportable.
- `IPCBuffer[T]` copies like a `Buffer` (`CopyFrom`/`CopyTo`, `DevicePtr` for
  `ArgDevicePtr`); `Close` unmaps this process's mapping
  (`cuIpcCloseMemHandle`) and never frees the exporter's memory.
- Events: create with `WithEventInterprocess()` (implies disabled timing),
  export with `(*Event).IPCHandle()`, and import with
  `OpenIPCEvent(ctx, h)`, which behaves like a timing-disabled `Event` for
  cross-process ordering. A plain event returns `ErrEventNotInterprocess`, and
  like the memory case the exporting event must stay open while any imported
  reference is in use.

The five IPC symbols are bound best-effort (`ErrSymbolUnavailable` on drivers
without them), and support varies by platform; the calls return the driver's
error where unsupported.

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

## managed (unified) memory

`ManagedBuffer` is unified memory addressable from both host and device: the
driver migrates pages on demand, so no explicit copy is needed. Write the host
`Slice`, launch a kernel against `DevicePtr`, and read the `Slice` back. It needs
a CUDA 6.0+ driver (prefetch and advise need 8.0+), so the entry points return
`ErrSymbolUnavailable` on older ones.

```go
mb, err := cuda.AllocManaged[float32](ctx, n)
if err != nil { return err }
defer mb.Close()
copy(mb.Slice(), input)                    // CPU writes directly
mb.PrefetchToDevice(context.Background(), stream)
fn.Launch(context.Background(), cfg, cuda.ArgDevicePtr(mb.DevicePtr()))
stream.Synchronize(context.Background())
result := mb.Slice()                        // CPU reads back, no explicit copy
```

- `func AllocManaged[T Supported](ctx *Context, n int) (*ManagedBuffer[T], error)`
  allocates `n` elements of unified memory (`cuMemAllocManaged`).
- `(*ManagedBuffer[T]).Slice() []T` returns a host-usable slice over the
  allocation; the CPU reads and writes it directly. Valid only while the buffer
  is open.
- `(*ManagedBuffer[T]).DevicePtr()` is the device pointer for kernel arguments
  (pass it with `ArgDevicePtr`); `Len`/`Bytes` report size.
- `(*ManagedBuffer[T]).PrefetchToDevice(ctx, stream)` / `PrefetchToHost(ctx, stream)`
  migrate the pages ahead of use (`cuMemPrefetchAsync`) so the first access does
  not page-fault.
- `(*ManagedBuffer[T]).Advise(advice MemAdvice)` applies a migration hint
  (`cuMemAdvise`), for example `AdviseSetReadMostly` or `AdviseSetPreferredLocation`.
- `(*ManagedBuffer[T]).Close()` frees the allocation (`cuMemFree`); close it
  before the Context.

## virtual memory (VMM)

`VirtualBuffer` is device memory built from the low-level virtual memory
management API: a reserved virtual address range with a physical allocation
mapped into it and device access granted. `AllocVirtual` wraps the whole
reserve, create, map, and set-access lifecycle (rolling back on any failure), so
the result behaves like a `Buffer` for launches and host copies. It is the
building block for custom growable allocators. The VMM symbols need a CUDA 10.2+
driver, so the entry points return `ErrSymbolUnavailable` on an older one.

```go
vb, err := cuda.AllocVirtual[float32](ctx, n)
if err != nil { return err }
defer vb.Close()
vb.CopyFrom(context.Background(), input)
fn.Launch(context.Background(), cfg, cuda.ArgDevicePtr(vb.DevicePtr()))
```

- `func AllocVirtual[T Supported](ctx *Context, n int) (*VirtualBuffer[T], error)`
  reserves and maps `n` elements, rounding the reservation up to the device's
  recommended granularity.
- `(*VirtualBuffer[T]).DevicePtr()` is the device pointer for kernel arguments;
  `Len`/`Bytes` report the requested size.
- `(*VirtualBuffer[T]).CopyFrom(ctx, src)` / `CopyTo(ctx, dst)` copy host to and
  from the buffer; lengths must match `Len()`.
- `(*VirtualBuffer[T]).Close()` tears the allocation down (unmap, release the
  handle, free the address reservation). It tears down only what is still live
  and is retryable: a partial failure returns an error and can be retried, and
  the address is freed only after the unmap succeeds.

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
its wait lane. That lane stays busy until the driver returns, but other waits
use separate lanes. Command submissions and queries on the context can
continue, while a later `Stream.Close` still waits for that accepted wait.

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
  `LogBufferBytes` (log buffer size, zero for the default; negative or over-cap
  values are rejected) and `MaxRegisters` (`CU_JIT_MAX_REGISTERS`, `0` leaves
  the driver default; values over `math.MaxUint32` are rejected). The simple
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

## JIT linking

`Linker` is a JIT link session that combines PTX and cubin inputs into a
single cubin image via `cuLink`, so separately compiled pieces can be linked
at runtime and then loaded like any other module. Like the other handle
types it is owned by a `Context` and locks its operations against a
concurrent `Close`.

```go
lk, err := ctx.NewLinker(cuda.JITOptions{})
if err != nil {
    log.Fatal(err)
}
defer lk.Close()

if err := lk.AddPTX("vector_add.ptx", ptx); err != nil {
    log.Fatalf("%v: %s", err, lk.Log().Error)
}
image, err := lk.Complete()
if err != nil {
    log.Fatalf("%v: %s", err, lk.Log().Error)
}
mod, err := ctx.LoadModule(image)
```

- `(*Context).NewLinker(opts JITOptions) (*Linker, error)` starts a link
  session. `JITOptions` is the same type `LoadModuleEx` uses: `LogBufferBytes`
  sizes the info and error log buffers (zero for the default; negative or
  over-cap values are rejected with `ErrInvalidLength`) and `MaxRegisters` caps
  registers per thread when `> 0` (values over `math.MaxUint32` are rejected
  with `ErrInvalidValue`). The driver retains the log buffers for the whole life of the link
  state, so they live on the `Linker` until `Close`.
- `(*Linker).AddPTX(name string, ptx []byte) error` adds a PTX input labelled
  `name` (empty is unlabelled). The image is null-terminated and its length is
  passed including the terminator, which the driver's PTX parser requires.
- `(*Linker).AddCubin(name string, cubin []byte) error` adds a cubin input
  with its exact bytes. Empty input is rejected with `ErrEmptyImage`.
- `(*Linker).Complete() ([]byte, error)` finishes the link and returns a fresh
  copy of the resulting cubin, ready for `LoadModule`. The driver owns the
  underlying buffer and frees it at `Close`, so the copy is taken before
  `Close` can run.
- `(*Linker).Log() JITLog` returns the info and error logs the driver produced,
  so a failed `AddPTX` or `Complete` keeps its diagnostics. A nil or
  never-created `Linker` returns the zero `JITLog`.
- `(*Linker).Close() error` destroys the session and the cubin buffer it owns.
  Idempotent after a successful destroy and safe on a nil receiver.

`cuLink` is a best-effort feature group (v2 entry points, CUDA 6.5+); a driver
missing any of its symbols makes `NewLinker` return `ErrSymbolUnavailable`. Add every input before calling
`Complete`, and close the `Linker` before its owning `Context`.

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
  Each call boxes its arguments (a few small allocations); for a loop that
  launches every iteration, prefer `Pack` + `LaunchPacked` below, which
  allocate nothing per launch.
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

### packed launches

`Launch` and `LaunchOn` build their argument array on every call, which boxes
each argument. For a tight loop that launches the same kernel repeatedly, pack
the arguments once and reuse them:

```go
p, err := cuda.Pack(cuda.Arg(in), cuda.Arg(out), cuda.ArgValue(int32(n)))
if err != nil {
    return err
}
for step := 0; step < steps; step++ {
    if err := fn.LaunchPacked(context.Background(), cfg, p); err != nil {
        return err
    }
}
```

- `Pack(args...) (*PackedArgs, error)` packs the same arguments `Launch` accepts
  into a reusable list. It resolves each buffer to a raw device pointer at pack
  time rather than holding the buffer's lock, so from then on the caller owns
  lifetime, like `ArgDevicePtr`.
- `(*Function).LaunchPacked(ctx, cfg, p)` and `LaunchPackedOn(ctx, stream, cfg, p)`
  launch with a pre-packed list and allocate nothing per launch.

**Lifetime rule:** a `PackedArgs` captures device pointers and texture and
surface handles, so keep every referenced `Buffer`, `Texture`, and `Surface`
(and the arrays they view) open and unchanged for as long as you launch it, and
do not copy a `PackedArgs` value (pass the pointer `Pack` returns). For one-off
launches, `Launch` is simpler and safer.

### cooperative launches

A cooperative kernel's blocks can synchronize as a whole grid (a grid-wide
barrier), which the hardware allows only when every block is co-resident on the
device at once.

```go
maxBlocks, err := fn.MaxCooperativeGridBlocks(256, 0) // co-resident ceiling
cfg := cuda.LaunchConfig1D(n, 256)
if int(cfg.GridX) <= maxBlocks {
    err = fn.LaunchCooperative(context.Background(), cfg, cuda.Arg(buf), cuda.ArgValue(int32(n)))
}
```

- `(*Function).LaunchCooperative(ctx, cfg, args...)` and
  `LaunchCooperativeOn(ctx, stream, cfg, args...)` launch through
  `cuLaunchCooperativeKernel`, taking the same arguments and lifetime rules as
  `Launch`. They return `ErrSymbolUnavailable` on a driver without the symbol.
- `(*Function).MaxCooperativeGridBlocks(blockSize, dynamicSharedMem int) (int, error)`
  returns the largest total block count that stays co-resident: the device's
  multiprocessor count times `MaxActiveBlocksPerSM`. The driver rejects a
  cooperative launch whose grid exceeds this with `ErrCooperativeLaunchTooLarge`.
  `DeviceAttributeCooperativeLaunch` reports whether the device supports
  cooperative launch at all.

## host functions on streams

`(*Stream).LaunchHostFunc(fn func()) error` enqueues a Go function into the
stream's order (`cuLaunchHostFunc`): the driver calls it on an internal thread
once all preceding stream work completes, and later stream work waits for it
to return.

```go
buf.CopyFromHostAsync(ctx, stream, host)
stream.LaunchHostFunc(func() { close(uploaded) })   // fires when the copy is done
fn.LaunchOn(ctx, stream, cfg, cuda.Arg(buf))
```

Rules: `fn` must not call back into CUDA and must not block on work from the
same stream. CUDA stalls the stream until it returns, and CUDA API calls from a
host function can deadlock. Keep it short, signal a channel, and return. Panics
inside `fn` are reported to stderr and swallowed because the caller is not a Go
thread. If the context is destroyed while host functions are still queued they
may never run, and their closures are retained for the life of the process. A
nil fn returns `ErrNilHostFunc`; a driver without the symbol returns
`ErrSymbolUnavailable`.

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
- `(*Array2D[T]).Raw() cudasys.CUarray` is the CUDA array handle.
- `(*Texture).Raw() cudasys.CUtexObject` is the texture object handle (the
  value a kernel receives as a `cudaTextureObject_t`).
- `(*Surface).Raw() cudasys.CUsurfObject` is the surface object handle (the
  value a kernel receives as a `cudaSurfaceObject_t`).

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
ErrLaunchFailed, ErrCooperativeLaunchTooLarge, ErrNotPermitted, ErrNotSupported, ErrSystemNotReady,
ErrSystemDriverMismatch, ErrStreamCaptureUnsupported, ErrStreamCaptureInvalidated,
ErrStreamCaptureMerge, ErrStreamCaptureUnmatched, ErrStreamCaptureUnjoined,
ErrStreamCaptureIsolation, ErrStreamCaptureImplicit, ErrCapturedEvent,
ErrStreamCaptureWrongThread, ErrTimeout, ErrGraphExecUpdateFailure,
ErrExternalDevice, ErrUnknown
```

Go-side sentinels:

- `ErrSymbolUnavailable`: an optional feature symbol (for example async allocation, graphs, virtual memory, or cooperative launch) was not present in the loaded driver, so that call cannot run. Core APIs are unaffected, so this is local to the feature rather than a load-time failure.
- `ErrCooperativeLaunchTooLarge`: a cooperative launch requested a grid larger than the device can keep co-resident. Size it with `MaxCooperativeGridBlocks`.
- `ErrInvalidOrdinal`: `GetDevice` rejected the ordinal before calling CUDA.
- `ErrNilDevice`: a method was called on a nil `*Device`.
- `ErrNilContext`: a method was called on a nil `*Context`.
- `ErrContextClosed`: a method was called on a `*Context` after `Close`.
- `ErrNilBuffer`: a buffer method received a nil buffer or typed-nil
  `PinnedHost[T]`.
- `ErrBufferClosed`: a buffer or pinned host region was used after `Close`.
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
- `ErrNilArray` / `ErrArrayClosed`: a method was called on a nil or closed `*Array2D[T]`.
- `ErrNilTexture` / `ErrTextureClosed`: a method was called on a nil or closed `*Texture`, or a closed texture was passed to `ArgTexture`.
- `ErrUnsupportedElement`: `AllocArray2D` was called with an 8-byte element type, which CUDA arrays do not support.
- `ErrNilHostFunc`: `Stream.LaunchHostFunc` was given a nil function.
- `ErrNilSurface` / `ErrSurfaceClosed`: a method was called on a nil or closed `*Surface`, or a closed surface was passed to `ArgSurface`.
- `ErrNoSurfaceStore`: `NewSurface` was given an array allocated without `WithSurfaceStore`.
- `ErrEventNotInterprocess`: `Event.IPCHandle` was called on an event created without `WithEventInterprocess`.

Returned CUDA errors for codes outside the table still match with:

```go
errors.Is(err, &cuda.Error{Code: code})
```
