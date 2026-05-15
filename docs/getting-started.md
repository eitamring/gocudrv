# getting started

`gocudrv` loads the NVIDIA CUDA driver at runtime. You do not need CUDA
headers, the CUDA toolkit, cgo, or a C compiler to build packages that import
it.

## requirements

- NVIDIA GPU with a working driver
- Linux, WSL2, or Windows
- Go 1.22+
- Precompiled PTX if you want to launch kernels

## WSL2

Install the NVIDIA driver on Windows. Do not install a Linux NVIDIA kernel
driver inside WSL. The Windows driver exposes CUDA to WSL through:

```text
/usr/lib/wsl/lib/libcuda.so.1
```

Basic sanity checks:

```bash
nvidia-smi
ls -l /usr/lib/wsl/lib/libcuda.so*
go version
```

## build and test

```bash
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go test ./...
go test -race ./...
```

Hardware-backed integration tests are opt-in:

```bash
go test -tags cuda_integration ./cuda
```

## device-info example

Run the current example to list visible CUDA devices:

```bash
go run ./examples/device-info
```

The command prints the driver version and per-device properties such as name,
total memory, compute capability, SM count, warp size, clock rate, and memory
bus width.
