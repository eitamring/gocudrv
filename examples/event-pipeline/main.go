// Command event-pipeline shows stream ordering with CUDA events.
//
// It runs two vector-add batches through two explicit streams. The copy stream
// uploads batch 1 while the compute stream can work on batch 0, then events
// order each handoff without synchronizing the whole context back to the CPU.
//
// Build:
//
//	go build ./examples/event-pipeline
//
// Regenerate the embedded PTX from ../vector-add/vector_add.cu (requires nvcc):
//
//	go generate ./examples/event-pipeline
//	# or
//	make ptx
package main

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/eitamring/gocudrv/cuda"
)

//go:generate bash ../vector-add/build-ptx.sh

//go:embed vector_add.ptx
var vectorAddPTX []byte

const (
	batches = 2
	n       = 1 << 20
	block   = 256
)

type workBatch struct {
	aHost *cuda.HostBuffer[float32]
	bHost *cuda.HostBuffer[float32]
	cHost *cuda.HostBuffer[float32]
	a     *cuda.Buffer[float32]
	b     *cuda.Buffer[float32]
	c     *cuda.Buffer[float32]
	ready *cuda.Event
	done  *cuda.Event
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "event-pipeline: %v\n", err)
		os.Exit(1)
	}
}

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

func cleanup(rerr *error, op string, fn func() error) {
	*rerr = addCleanupErr(*rerr, op, fn())
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
	defer cleanup(&rerr, "context close", ctx.Close)

	copyStream, err := ctx.NewStream()
	if err != nil {
		return fmt.Errorf("copy stream: %w", err)
	}
	defer cleanup(&rerr, "copy stream close", copyStream.Close)

	computeStream, err := ctx.NewStream()
	if err != nil {
		return fmt.Errorf("compute stream: %w", err)
	}
	defer cleanup(&rerr, "compute stream close", computeStream.Close)

	start, err := ctx.NewEvent()
	if err != nil {
		return fmt.Errorf("start event: %w", err)
	}
	defer cleanup(&rerr, "start event close", start.Close)

	stop, err := ctx.NewEvent()
	if err != nil {
		return fmt.Errorf("stop event: %w", err)
	}
	defer cleanup(&rerr, "stop event close", stop.Close)

	mod, err := ctx.LoadModule(vectorAddPTX)
	if err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	defer cleanup(&rerr, "module close", mod.Close)

	fn, err := mod.Function("vector_add")
	if err != nil {
		return fmt.Errorf("function lookup: %w", err)
	}

	var batch [batches]*workBatch
	for i := range batch {
		batch[i], err = newBatch(ctx, i)
		if err != nil {
			return err
		}
		defer batch[i].close(&rerr, i)
	}

	bg := context.Background()
	submitted := false
	defer func() {
		if submitted {
			rerr = addCleanupErr(rerr, "synchronize before cleanup", ctx.Synchronize(bg))
		}
	}()

	if err := start.Record(copyStream); err != nil {
		return fmt.Errorf("record start: %w", err)
	}
	for i, b := range batch {
		if err := b.a.CopyFromHostAsync(bg, copyStream, b.aHost); err != nil {
			return fmt.Errorf("batch %d copy a: %w", i, err)
		}
		submitted = true
		if err := b.b.CopyFromHostAsync(bg, copyStream, b.bHost); err != nil {
			return fmt.Errorf("batch %d copy b: %w", i, err)
		}
		if err := b.ready.Record(copyStream); err != nil {
			return fmt.Errorf("batch %d record ready: %w", i, err)
		}
		if err := computeStream.WaitEvent(b.ready); err != nil {
			return fmt.Errorf("batch %d wait ready: %w", i, err)
		}
		if err := fn.LaunchOn(bg, computeStream, cuda.LaunchConfig1D(n, block),
			cuda.Arg(b.a),
			cuda.Arg(b.b),
			cuda.Arg(b.c),
			cuda.ArgValue(int32(n)),
		); err != nil {
			return fmt.Errorf("batch %d launch: %w", i, err)
		}
		if err := b.done.Record(computeStream); err != nil {
			return fmt.Errorf("batch %d record done: %w", i, err)
		}
	}
	for i, b := range batch {
		if err := copyStream.WaitEvent(b.done); err != nil {
			return fmt.Errorf("batch %d wait done: %w", i, err)
		}
		if err := b.c.CopyToHostAsync(bg, copyStream, b.cHost); err != nil {
			return fmt.Errorf("batch %d copy result: %w", i, err)
		}
	}
	if err := stop.Record(copyStream); err != nil {
		return fmt.Errorf("record stop: %w", err)
	}

	query := "pending"
	if err := stop.Query(); err == nil {
		query = "ready"
	} else if !errors.Is(err, cuda.ErrNotReady) {
		return fmt.Errorf("query stop: %w", err)
	}

	if err := stop.Synchronize(bg); err != nil {
		return fmt.Errorf("stop synchronize: %w", err)
	}
	submitted = false

	elapsed, err := start.Elapsed(stop)
	if err != nil {
		return fmt.Errorf("elapsed: %w", err)
	}
	for i, b := range batch {
		if err := verifyBatch(i, b); err != nil {
			return err
		}
	}

	fmt.Printf("ok batches=%d n=%d gpu_time=%s stop_query_before_sync=%s\n",
		batches, n, formatDuration(elapsed), query)
	return nil
}

func newBatch(ctx *cuda.Context, batchIndex int) (b *workBatch, err error) {
	b = &workBatch{}
	defer func() {
		if err == nil {
			return
		}
		var cleanupErr error
		b.close(&cleanupErr, batchIndex)
		err = errors.Join(err, cleanupErr)
	}()
	b.aHost, err = cuda.AllocHost[float32](ctx, n)
	if err != nil {
		return nil, fmt.Errorf("batch %d alloc host a: %w", batchIndex, err)
	}
	b.bHost, err = cuda.AllocHost[float32](ctx, n)
	if err != nil {
		return nil, fmt.Errorf("batch %d alloc host b: %w", batchIndex, err)
	}
	b.cHost, err = cuda.AllocHost[float32](ctx, n)
	if err != nil {
		return nil, fmt.Errorf("batch %d alloc host c: %w", batchIndex, err)
	}
	b.a, err = cuda.Alloc[float32](ctx, n)
	if err != nil {
		return nil, fmt.Errorf("batch %d alloc device a: %w", batchIndex, err)
	}
	b.b, err = cuda.Alloc[float32](ctx, n)
	if err != nil {
		return nil, fmt.Errorf("batch %d alloc device b: %w", batchIndex, err)
	}
	b.c, err = cuda.Alloc[float32](ctx, n)
	if err != nil {
		return nil, fmt.Errorf("batch %d alloc device c: %w", batchIndex, err)
	}
	b.ready, err = ctx.NewEvent(cuda.WithEventDisableTiming())
	if err != nil {
		return nil, fmt.Errorf("batch %d ready event: %w", batchIndex, err)
	}
	b.done, err = ctx.NewEvent(cuda.WithEventDisableTiming())
	if err != nil {
		return nil, fmt.Errorf("batch %d done event: %w", batchIndex, err)
	}

	a := b.aHost.Slice()
	bb := b.bHost.Slice()
	base := float32(batchIndex * n)
	for i := range a {
		a[i] = base + float32(i)
		bb[i] = float32(i) * 0.5
	}
	return b, nil
}

func (b *workBatch) close(rerr *error, index int) {
	if b.done != nil {
		cleanup(rerr, fmt.Sprintf("batch %d done event close", index), b.done.Close)
	}
	if b.ready != nil {
		cleanup(rerr, fmt.Sprintf("batch %d ready event close", index), b.ready.Close)
	}
	if b.c != nil {
		cleanup(rerr, fmt.Sprintf("batch %d device c close", index), b.c.Close)
	}
	if b.b != nil {
		cleanup(rerr, fmt.Sprintf("batch %d device b close", index), b.b.Close)
	}
	if b.a != nil {
		cleanup(rerr, fmt.Sprintf("batch %d device a close", index), b.a.Close)
	}
	if b.cHost != nil {
		cleanup(rerr, fmt.Sprintf("batch %d host c close", index), b.cHost.Close)
	}
	if b.bHost != nil {
		cleanup(rerr, fmt.Sprintf("batch %d host b close", index), b.bHost.Close)
	}
	if b.aHost != nil {
		cleanup(rerr, fmt.Sprintf("batch %d host a close", index), b.aHost.Close)
	}
}

func verifyBatch(batchIndex int, b *workBatch) error {
	a := b.aHost.Slice()
	bb := b.bHost.Slice()
	c := b.cHost.Slice()
	for i := range c {
		want := a[i] + bb[i]
		if c[i] != want {
			return fmt.Errorf("batch %d mismatch at %d: got %v, want %v", batchIndex, i, c[i], want)
		}
	}
	return nil
}

func formatDuration(d time.Duration) string {
	if d < time.Microsecond {
		return d.String()
	}
	return d.Round(time.Microsecond).String()
}
