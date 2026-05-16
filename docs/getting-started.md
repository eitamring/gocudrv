# getting started

`gocudrv` loads the CUDA driver at runtime. Building it does not require the
CUDA toolkit, CUDA headers, cgo, or a C compiler.

## requirements

- NVIDIA GPU with a working driver
- Linux, WSL2, or Windows
- Go 1.22+
- Precompiled PTX if you want to launch kernels

## WSL2

Install the NVIDIA driver on Windows. Do not install a Linux NVIDIA kernel
driver inside WSL. CUDA should be exposed through:

```text
/usr/lib/wsl/lib/libcuda.so.1
```

Sanity check:

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

## examples

List visible devices:

```bash
go run ./examples/device-info
```

Run the vector-add example:

```bash
go run ./examples/vector-add
```

For the `.cu` to `.ptx` workflow, see [writing and shipping kernels](kernels.md).
