# gocudrv

Pure-Go bindings for the NVIDIA CUDA Driver API. No cgo. The driver library is
loaded dynamically at runtime.

Status: very early. The current API covers initialization, device discovery,
primary contexts, memory, module loading, and default-stream kernel launch.

## What it is

A thin Go wrapper around `libcuda.so.1` / `nvcuda.dll` so a Go program can:

- init CUDA
- enumerate devices
- create primary contexts
- allocate device memory
- copy memory
- load precompiled PTX
- launch kernels
- record events

All without `cgo`, a C compiler, or the CUDA toolkit being installed at build
time.

## How it works

At runtime, `gocudrv` opens the CUDA driver library (`libcuda.so.1` on Linux /
WSL2, `nvcuda.dll` on Windows), binds the driver API symbols with pure Go, and
wraps the raw CUDA result codes as Go errors. The public `cuda` package keeps
raw handles behind Go types such as `Device`.

## Requirements

- NVIDIA GPU with a working driver
- Linux, WSL2, or Windows
- Go 1.22+
- precompiled PTX if you want to launch kernels

CUDA headers and the CUDA toolkit are not required to build this package.

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

## Docs

- [Getting started](docs/getting-started.md)
- [Writing and shipping kernels](docs/kernels.md)
- [Public API](docs/api/cuda.md)
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
7. events and basic benchmarking

## License

See `LICENSE`.
