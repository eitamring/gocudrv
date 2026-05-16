// Command vector-add runs the canonical CUDA hello-world: add two float
// vectors elementwise on the GPU and verify the result on the CPU.
//
// The PTX is embedded into the binary, so the resulting executable has
// no runtime dependency on the .ptx file or the CUDA toolkit. It still
// requires a working NVIDIA driver.
//
// Build:
//
//	go build ./examples/vector-add
//
// Regenerate the embedded PTX from vector_add.cu (requires nvcc):
//
//	go generate ./examples/vector-add
//	# or
//	make ptx
//	# or
//	bash examples/vector-add/build-ptx.sh
//
// All three regeneration paths invoke the same script so the output is
// identical and the cuda package's integration fixture stays in sync.
package main

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"

	"github.com/eitamring/gocudrv/cuda"
)

//go:generate bash build-ptx.sh

//go:embed vector_add.ptx
var vectorAddPTX []byte

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "vector-add: %v\n", err)
		os.Exit(1)
	}
}

// addCleanupErr folds err from a deferred cleanup into the operational
// error. When both are non-nil the result joins them so the caller sees
// both the original failure and the cleanup failure rather than only
// one. This is the pattern the SDK expects from production code.
func addCleanupErr(prev error, op string, err error) error {
	if err == nil {
		return prev
	}
	wrapped := fmt.Errorf("%s: %w", op, err)
	if prev == nil {
		return wrapped
	}
	return errors.Join(prev, wrapped)
}

func run() (rerr error) {
	if err := cuda.Init(); err != nil {
		return fmt.Errorf("init: %w", err)
	}

	dev, err := cuda.GetDevice(0)
	if err != nil {
		return fmt.Errorf("device: %w", err)
	}
	name, err := dev.Name()
	if err != nil {
		return fmt.Errorf("device name: %w", err)
	}
	fmt.Printf("device: %s\n", name)

	ctx, err := dev.Primary()
	if err != nil {
		return fmt.Errorf("primary context: %w", err)
	}
	defer func() { rerr = addCleanupErr(rerr, "context close", ctx.Close()) }()

	const n = 1024
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	for i := range aHost {
		aHost[i] = float32(i)
		bHost[i] = float32(i) * 2
	}

	a, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		return fmt.Errorf("alloc a: %w", err)
	}
	defer func() { rerr = addCleanupErr(rerr, "buffer a close", a.Close()) }()

	b, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		return fmt.Errorf("alloc b: %w", err)
	}
	defer func() { rerr = addCleanupErr(rerr, "buffer b close", b.Close()) }()

	c, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		return fmt.Errorf("alloc c: %w", err)
	}
	defer func() { rerr = addCleanupErr(rerr, "buffer c close", c.Close()) }()

	bg := context.Background()
	if err := a.CopyFrom(bg, aHost); err != nil {
		return fmt.Errorf("copy a: %w", err)
	}
	if err := b.CopyFrom(bg, bHost); err != nil {
		return fmt.Errorf("copy b: %w", err)
	}

	mod, err := ctx.LoadModule(vectorAddPTX)
	if err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	defer func() { rerr = addCleanupErr(rerr, "module close", mod.Close()) }()

	fn, err := mod.Function("vector_add")
	if err != nil {
		return fmt.Errorf("function lookup: %w", err)
	}

	if err := fn.Launch(bg, cuda.LaunchConfig1D(n, 256),
		cuda.Arg(a),
		cuda.Arg(b),
		cuda.Arg(c),
		cuda.ArgValue(int32(n)),
	); err != nil {
		return fmt.Errorf("launch: %w", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		return fmt.Errorf("synchronize: %w", err)
	}

	got := make([]float32, n)
	if err := c.CopyTo(bg, got); err != nil {
		return fmt.Errorf("copy result: %w", err)
	}

	for i := range got {
		want := aHost[i] + bHost[i]
		if got[i] != want {
			return fmt.Errorf("mismatch at %d: got %v, want %v", i, got[i], want)
		}
	}
	fmt.Printf("ok n=%d, c[0]=%v, c[%d]=%v\n", n, got[0], n-1, got[n-1])
	return nil
}
