# gocudrv

Pure-Go bindings for the NVIDIA CUDA Driver API. No cgo. The driver library is
loaded dynamically at runtime.

Status: the core Driver API surface is in place and tested. Unit tests run
without a GPU (CGO_ENABLED=0), and the API is not yet frozen. The current API
covers initialization, device discovery,
primary contexts, memory, module loading, kernel launch, explicit streams,
events, async pinned copies, device memory primitives (memset, typed fill,
device-to-device copy, free/total query), buffer subrange (offset) copies,
non-owning buffer views, host memory registration, flat and shaped async copies
with pinned memory, pitched and volume allocations, CUDA arrays, memory pool
access, non-blocking stream polling, module JIT options and logs, raw and unsafe
kernel arguments, occupancy helpers, device global access, CUDA graph capture,
replay, and update, stream-ordered async allocation,
JIT linking (cuLink), device diagnostics (PCI bus id, UUID, and more attributes),
and raw handle accessors for sibling-module integration.

## What it is

A thin Go wrapper around `libcuda.so.1` / `nvcuda.dll` so a Go program can:

- init CUDA
- enumerate devices
- create primary contexts
- allocate device memory
- copy memory
- zero buffers and copy device-to-device
- query free and total device memory
- load precompiled PTX
- launch kernels
- create and synchronize streams
- record/query events and wait across streams
- enqueue flat and shaped async copies with allocated or registered pinned memory

All without `cgo`, a C compiler, or the CUDA toolkit being installed at build
time.

## How it works

At runtime, `gocudrv` opens the CUDA driver library (`libcuda.so.1` on Linux /
WSL2, `nvcuda.dll` on Windows), binds the driver API symbols with pure Go, and
wraps the raw CUDA result codes as Go errors. The public `cuda` package keeps
raw handles behind Go types such as `Device`.

## Scope

`gocudrv` is a focused CUDA Driver API wrapper. Higher-level stacks (OCR,
inference, image decode, model orchestration) belong in separate modules layered
*above* this one. A sibling module can reuse this package's context, streams, and
buffers through the raw handle accessors; see
[sibling module integration](docs/integration.md) for the contract.

## Requirements

- NVIDIA GPU with a working driver
- Linux, WSL2, or Windows
- Go 1.24+
- precompiled PTX if you want to launch kernels

CUDA headers and the CUDA toolkit are not required to build this package.

The core driver symbols are required at init and have been stable across many
CUDA releases, so the package loads on a wide range of drivers. The feature
groups (stream-ordered allocation, CUDA graphs, occupancy, and device
diagnostics) are bound best-effort: on a driver that lacks one, `Init` still
succeeds, and only the matching call returns `ErrSymbolUnavailable`. Async
allocation needs CUDA 11.2 and graphs need a CUDA 11.x driver when used. See
`docs/internals.md` for the full symbol table and the per-group breakdown.

## WSL2 quickstart

Install the NVIDIA driver on Windows. Do not install a Linux NVIDIA kernel driver inside WSL. CUDA is exposed to WSL through `/usr/lib/wsl/lib/libcuda.so.1`.

Sanity check:

```bash
nvidia-smi
ls -l /usr/lib/wsl/lib/libcuda.so*
go version
```

## Build

```bash
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go test ./...
```

## Try it

```bash
go run ./examples/vector-add
```

Adds two `float32` vectors on the GPU using an embedded PTX module and
verifies the result on the CPU. See `docs/kernels.md` for the full
workflow from `.cu` source to a running Go program.

For streams, events, and async pinned copies:

```bash
go run ./examples/event-pipeline
```

This runs two vector-add batches through a small event-ordered pipeline and
prints GPU elapsed time from CUDA events.

## Benchmarks

```bash
go test -bench . ./cuda                         # wrapper and executor overhead (no GPU)
go test -tags cuda_integration -bench . ./cuda # real bandwidth and GPU latency (needs a GPU)
```

The default benchmarks run against a fake driver, so they measure gocudrv's own
CPU enqueue costs, including launch latency. The `cuda_integration` benchmarks
measure real copy bandwidth, small-kernel end-to-end latency, bidirectional copy
overlap, and memset throughput. GPU benchmarks report event-timed metrics
alongside wall time so device work can be read apart from CPU submission. None
of them need model or application assets.

## Docs

- [Getting started](docs/getting-started.md)
- [Writing and shipping kernels](docs/kernels.md)
- [Public API](docs/api/cuda.md)
- [Sibling module integration](docs/integration.md)
- [Internals](docs/internals.md)
- [Docs index](docs/README.md)

## Layout

```
cuda/          public API, tests, and benchmarks
cudaresult/    CUDA result-to-error wrappers
cudasys/       raw Driver API types and dynamic symbols
internal/      loader, executor, arg packing, platform paths, and host callbacks
docs/          guides, API reference, and internals
examples/      runnable examples
scripts/       build and check helpers
```

## Roadmap

The initial roadmap is complete. Follow-up work is tracked in GitHub issues and small, focused pull requests.

## License

See `LICENSE`.
