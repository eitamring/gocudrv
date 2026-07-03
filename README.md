# gocudrv

Pure-Go bindings for the NVIDIA CUDA Driver API. No cgo. The driver library is
loaded dynamically at runtime.

Status: very early. The current API covers initialization, device discovery,
primary contexts, memory, module loading, kernel launch, explicit streams,
events, async pinned copies, device memory primitives (memset, typed fill,
device-to-device copy, free/total query), buffer subrange (offset) copies,
non-owning buffer views, host memory registration, pitched 2D allocation and
copies, memory pool access, non-blocking stream polling, module JIT options and
logs, raw and unsafe kernel arguments, occupancy helpers, device global access, CUDA graph capture, replay, and update, stream-ordered async allocation,
device diagnostics (PCI bus id, UUID, and more attributes), and raw handle
accessors for sibling-module integration.

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
- enqueue async pinned-memory copies

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
go test -bench . ./cuda                      # wrapper and executor overhead (no GPU)
go test -tags cuda_integration -bench . ./cuda  # real bandwidth and launch overhead (needs a GPU)
```

The default benchmarks run against a fake driver, so they measure gocudrv's own
CPU enqueue cost (executor round trip, locking, argument packing). The
`cuda_integration` benchmarks measure real device-copy bandwidth and launch
overhead; `BenchmarkRealAsyncPinnedCopy` reports a `gpu-us/op` metric measured
with CUDA events alongside the wall-time `ns/op`, so the GPU transfer time can
be read apart from the CPU submit time. None of them need OCR assets or model
files.

## Docs

- [Getting started](docs/getting-started.md)
- [Writing and shipping kernels](docs/kernels.md)
- [Public API](docs/api/cuda.md)
- [Sibling module integration](docs/integration.md)
- [Internals](docs/internals.md)
- [Docs index](docs/README.md)

## Layout

```
cudasys/       raw dynamic symbols, close to C ABI
cudaresult/    thin wrappers returning Go errors
cuda/          public Go API
internal/      dynamic loader, executor, arg packing, platform paths
examples/      runnable demos
scripts/       build and check helpers
```

## Roadmap

1. dynamic driver loader and `cuInit`
2. device enumeration
3. context with pinned executor goroutine
4. device memory and host/device copies
5. PTX module loading
6. kernel launch with arg packing
7. streams
8. async host/device copies
9. events and stream waits
10. device memory primitives: memset, typed fill, device-to-device copy, free/total query
11. occupancy helpers and device global access
12. basic benchmarking: bandwidth, launch latency, overlap, memset throughput
13. CUDA graph capture and replay
14. stream-ordered async allocation

## License

See `LICENSE`.
