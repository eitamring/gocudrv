# Sibling module integration

This page is the contract a future GPU module built on top of `gocudrv` (for
example a `gocudrv-cublas`, `gocudrv-cudnn`, `gocudrv-tensorrt`, or
`gocudrv-nvjpeg`) can depend on. It defines how such a module shares this
package's CUDA resources without forking it, and what stays its own
responsibility.

`gocudrv` stays a focused CUDA Driver API wrapper. OCR, inference, image decode,
model orchestration, and file ingestion belong in separate modules layered
*above* this one, not in this repo.

## Raw handle ownership

A sibling module reuses `gocudrv`'s resources through the raw handle accessors
(see [raw handles](api/cuda.md#raw-handles-advanced)):

- `(*Context).Raw()` and `(*Context).Driver()` for the primary context and the
  loaded driver,
- `(*Stream).Raw()` and `(*Event).Raw()` for stream and event handles,
- `(*Buffer[T]).DevicePtr()` for a device pointer.

`gocudrv` owns every handle it hands out. A sibling **borrows** them: it must not
destroy or free a handle it received (no `cuStreamDestroy`, `cuMemFree`, and so
on on a borrowed handle), and must not use a handle after its owning Go value is
closed. Lifetime is the owner's; the sibling holds a non-owning reference.

## Context and stream sharing

Share one `Context` and pass its streams to the sibling rather than creating a
second context per device. CUDA's current context is per OS thread, and
`gocudrv` keeps context-affine work on a pinned executor goroutine. A sibling
that issues its own driver calls on its own goroutine must make the right
context current on that thread first (via `Context.Driver()` and
`cuCtxSetCurrent`), or run on a thread where it already is. Reusing
`Context.Driver()` also avoids loading `libcuda` a second time.

Order work across the shared streams with events (`Stream.WaitEvent`,
`Event.Record`) instead of host synchronization, so the sibling's kernels and
`gocudrv`'s copies interleave on the same timeline.

## Buffer pointer lifetime

A `Buffer[T].DevicePtr()` is valid only while that `Buffer` is open. Pair every
device pointer handed to a sibling with the `Buffer` that owns it, keep the
`Buffer` alive for as long as the sibling may touch the pointer, and do not call
`Close` or `FreeAsync` until the sibling's work on that memory has completed
(synchronize the stream first). The pointer is a raw snapshot; `gocudrv` cannot
detect use after free.

## Optional symbols and versions

Feature groups (async allocation, occupancy, graphs, device diagnostics, host
registration, pitched memory, memory pools) are bound best-effort, so on an
older driver the matching call returns `ErrSymbolUnavailable` rather than failing
at init. A sibling that depends on one of these should check for
`ErrSymbolUnavailable` and degrade, and should state its own minimum driver
version. The core init, device, context, memory, module, launch, stream, and
event symbols are always present once `Init` succeeds.

## Example

`examples/sibling-module` shows the handoff end to end: it acquires a context,
stream, and buffers from `gocudrv`, passes the raw handles to a stand-in for a
foreign library entry point, and synchronizes. It imports no real external CUDA
library, so it stays a contract demonstration rather than a dependency.
