// Command sibling-module shows how a module built on top of gocudrv (a future
// gocudrv-cublas, gocudrv-tensorrt, and so on) reuses a gocudrv context, stream,
// and buffers through the raw handle accessors instead of opening its own
// driver. It imports no real external CUDA library; foreignKernel stands in for
// one.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudasys"
)

// foreignKernel stands in for an entry point in another CUDA library (for
// example cublasSgemm). A real one would call into that library with these
// borrowed handles; here it only reports the handoff. It must not free or close
// any of them, and must not use them past the owner's Close.
func foreignKernel(drv *cudasys.Driver, stream cudasys.CUstream, in, out cudasys.CUdeviceptr) error {
	if drv == nil {
		return fmt.Errorf("nil driver")
	}
	fmt.Printf("sibling received stream=%#x in=%#x out=%#x\n", stream, in, out)
	return nil
}

func main() {
	if err := cuda.Init(); err != nil {
		fail("init", err)
	}
	dev, err := cuda.GetDevice(0)
	if err != nil {
		fail("device", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		fail("primary", err)
	}
	defer ctx.Close()

	stream, err := ctx.NewStream()
	if err != nil {
		fail("stream", err)
	}
	defer stream.Close()

	in, err := cuda.Alloc[float32](ctx, 1024)
	if err != nil {
		fail("alloc in", err)
	}
	defer in.Close()
	out, err := cuda.Alloc[float32](ctx, 1024)
	if err != nil {
		fail("alloc out", err)
	}
	defer out.Close()

	// Hand the borrowed handles to the sibling. gocudrv keeps ownership; the
	// sibling must not close them, and the buffers must outlive its work.
	if err := foreignKernel(ctx.Driver(), stream.Raw(), in.DevicePtr(), out.DevicePtr()); err != nil {
		fail("sibling", err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		fail("synchronize", err)
	}
	fmt.Println("ok")
}

func fail(op string, err error) {
	fmt.Fprintf(os.Stderr, "sibling-module: %s: %v\n", op, err)
	os.Exit(1)
}
