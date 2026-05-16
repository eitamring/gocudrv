// vector_add.cu adds two float vectors elementwise.
//
// Regenerate the PTX in this repo with the canonical script (which also
// keeps cuda/testdata in sync and uses -arch=sm_50):
//
//   bash build-ptx.sh
//   # or, from the repo root:
//   make ptx
//   go generate ./examples/vector-add
//
// extern "C" disables C++ name mangling so the kernel symbol is named
// "vector_add" exactly, which is what the Go side passes to Module.Function.
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
