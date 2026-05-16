# writing and shipping kernels

`gocudrv` loads GPU code; it does not compile it. Write kernels in CUDA C,
compile them with `nvcc`, then load the PTX from Go.

```text
vector_add.cu -> nvcc -> vector_add.ptx -> LoadModule -> Function -> Launch
```

You need the CUDA toolkit to build PTX. You do not need it to build or run the
Go binary once the PTX already exists.

## kernel source

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

Use `extern "C"` if you want to look the function up as
`mod.Function("vector_add")`. Without it, C++ name mangling changes the symbol
name.

## build PTX

```bash
nvcc -ptx -arch=sm_50 vector_add.cu -o vector_add.ptx
```

`sm_50` is a conservative baseline for the example. Use a newer target if your
kernel needs newer GPU features.

## load it from Go

The common path is to embed the PTX:

```go
import _ "embed"

//go:generate nvcc -ptx -arch=sm_50 vector_add.cu -o vector_add.ptx

//go:embed vector_add.ptx
var vectorAddPTX []byte

mod, err := ctx.LoadModule(vectorAddPTX)
```

If you want to ship the PTX as a file instead:

```go
mod, err := ctx.LoadModuleFromFile("./kernels/vector_add.ptx")
```

## this repo

The vector-add example keeps both the `.cu` source and checked-in PTX:

```bash
make ptx
# or
bash examples/vector-add/build-ptx.sh
# or
go generate ./examples/vector-add
```

Those commands all use the same script. It also keeps
`cuda/testdata/vector_add.ptx` in sync with the example fixture.

Run the example with:

```bash
go run ./examples/vector-add
```

It uses normal Go slices for the shortest path through the API. For larger or
repeated transfers, use `cuda.AllocHost` with `CopyFromHost` / `CopyToHost`.
