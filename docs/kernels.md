# writing and shipping kernels

`gocudrv` is a host-side binding. It loads precompiled CUDA modules and
launches their kernels; it does not compile CUDA source. You write
kernels in CUDA C, compile them once with `nvcc`, and ship the resulting
PTX (or cubin) with your Go program.

## what the toolchain looks like

```text
vector_add.cu  --nvcc-->  vector_add.ptx  --LoadModule-->  Function  --Launch-->  GPU
```

The CUDA toolkit (`nvcc`) is needed only to produce PTX. Building
`gocudrv` or programs that use it does not require the toolkit. CI
builds without it. End users running your compiled binary do not need
it either; they only need a working NVIDIA driver.

## writing a kernel

```cuda
extern "C" __global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

`extern "C"` disables C++ name mangling so the symbol the Go side asks
for (`mod.Function("vector_add")`) matches exactly. Without it the name
becomes something like `_Z10vector_addPKfS0_Pfi`.

## compiling to PTX

```bash
nvcc -ptx -arch=sm_50 vector_add.cu -o vector_add.ptx
```

`-ptx` stops after generating PTX (no SASS, no linking). `-arch=sm_50`
targets a baseline compute capability so the resulting PTX works on
every GPU from Maxwell onward; the driver JIT-compiles PTX to the
actual SASS at load time. Pick a higher arch if you depend on features
introduced later (e.g., `sm_70` for tensor cores).

## loading the PTX in Go

Two patterns, depending on whether you want the PTX bundled into the
binary or loaded from disk.

### pattern 1: embed with `//go:embed`

The resulting binary has no external file dependency. Best for
distributed programs.

```go
package main

import (
    _ "embed"
    "github.com/eitamring/gocudrv/cuda"
)

//go:generate nvcc -ptx -arch=sm_50 vector_add.cu -o vector_add.ptx

//go:embed vector_add.ptx
var vectorAddPTX []byte

func main() {
    cuda.Init()
    dev, _ := cuda.GetDevice(0)
    ctx, _ := dev.Primary()
    defer ctx.Close()

    mod, _ := ctx.LoadModule(vectorAddPTX)
    defer mod.Close()
    // ...
}
```

The `//go:embed` directive embeds the file at compile time; it must
live in the same package directory as the source file that uses it.
The `//go:generate` line is optional; it lets `go generate ./...`
regenerate the PTX when `vector_add.cu` changes (and the developer
has `nvcc` available). Keep `-arch=sm_50` consistent with the
checked-in artifact so regenerations target the same compatibility
baseline. The driver still JIT-compiles PTX to device code at load
time for the actual GPU, so the final SASS can differ across drivers,
toolchains, and hardware; only the PTX-level baseline is what `-arch`
locks down.

In this repo, the `examples/vector-add/` directive runs
`bash build-ptx.sh` instead of `nvcc` directly. The script bundles
the same `nvcc` invocation and also keeps the integration-test
fixture in `cuda/testdata/` byte-identical, which a unit test
guards. For your own project the direct `nvcc` line above is fine.

### pattern 2: load from disk

When the PTX is shipped alongside the binary or is generated at
deploy time.

```go
mod, err := ctx.LoadModuleFromFile("./kernels/vector_add.ptx")
```

`LoadModuleFromFile` is sugar for reading the file with `os.ReadFile`
and calling `LoadModule`.

## regenerating PTX in this repo

The `examples/vector-add/` directory ships both the `.cu` source and a
checked-in PTX. To regenerate after editing the source:

```bash
make ptx
# or directly
bash examples/vector-add/build-ptx.sh
# or via go generate
go generate ./examples/vector-add
```

Any of these requires `nvcc` on `PATH`. None of them runs in CI: the
checked-in PTX is the source of truth for the example.

## end-to-end example

See `examples/vector-add/` for a runnable program:

```bash
go run ./examples/vector-add
```

Output on a working GPU:

```text
device: NVIDIA GeForce RTX 4070 Ti
ok n=1024, c[0]=0, c[1023]=3069
```

The example uses the simplest copy path: pageable Go slices via
`Buffer.CopyFrom` / `CopyTo`. This keeps the hello-world short. For
real workloads with repeated or large transfers, use `cuda.AllocHost`
and `Buffer.CopyFromHost` / `CopyToHost` instead; pinned host memory
lets CUDA DMA directly to and from your slice without an internal
staging copy. See the "pinned host memory" section in the public API
docs.

The same kernel is used by `TestRealVectorAddLaunch` under the
`cuda_integration` build tag. A unit test guards against the two PTX
files (the example's embedded copy and the test fixture) drifting; the
regeneration script writes both at once.
