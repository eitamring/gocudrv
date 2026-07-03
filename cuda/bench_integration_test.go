//go:build cuda_integration

package cuda

import (
	"context"
	"testing"
	"time"
)

func benchRealContext(b *testing.B) *Context {
	b.Helper()
	if err := Init(); err != nil {
		b.Skipf("CUDA driver is not usable in this environment: %v", err)
	}
	dev, err := GetDevice(0)
	if err != nil {
		b.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		b.Fatalf("Primary: %v", err)
	}
	b.Cleanup(func() { _ = ctx.Close() })
	return ctx
}

const benchCopyElems = 1 << 20 // 4 MiB of float32

func BenchmarkRealPageableCopy(b *testing.B) {
	ctx := benchRealContext(b)
	buf, err := Alloc[float32](ctx, benchCopyElems)
	if err != nil {
		b.Fatalf("Alloc: %v", err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	src := make([]float32, benchCopyElems)
	b.SetBytes(benchCopyElems * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyFrom(context.Background(), src); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRealPinnedCopy(b *testing.B) {
	ctx := benchRealContext(b)
	buf, err := Alloc[float32](ctx, benchCopyElems)
	if err != nil {
		b.Fatalf("Alloc: %v", err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	host, err := AllocHost[float32](ctx, benchCopyElems)
	if err != nil {
		b.Fatalf("AllocHost: %v", err)
	}
	b.Cleanup(func() { _ = host.Close() })
	b.SetBytes(benchCopyElems * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyFromHost(context.Background(), host); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRealAsyncPinnedCopy reports both the default wall time per op (CPU
// submit plus the synchronize) and a gpu-us/op metric measured with events, so
// the CPU enqueue cost can be read apart from the GPU transfer time.
func BenchmarkRealAsyncPinnedCopy(b *testing.B) {
	ctx := benchRealContext(b)
	stream, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream: %v", err)
	}
	b.Cleanup(func() { _ = stream.Close() })
	buf, err := Alloc[float32](ctx, benchCopyElems)
	if err != nil {
		b.Fatalf("Alloc: %v", err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	host, err := AllocHost[float32](ctx, benchCopyElems)
	if err != nil {
		b.Fatalf("AllocHost: %v", err)
	}
	b.Cleanup(func() { _ = host.Close() })
	start, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent: %v", err)
	}
	b.Cleanup(func() { _ = start.Close() })
	done, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent: %v", err)
	}
	b.Cleanup(func() { _ = done.Close() })

	bg := context.Background()
	b.SetBytes(benchCopyElems * 4)
	b.ResetTimer()
	var gpuUS float64
	for i := 0; i < b.N; i++ {
		_ = start.Record(stream)
		if err := buf.CopyFromHostAsync(bg, stream, host); err != nil {
			b.Fatal(err)
		}
		_ = done.Record(stream)
		if err := stream.Synchronize(bg); err != nil {
			b.Fatal(err)
		}
		d, err := start.Elapsed(done)
		if err != nil {
			b.Fatal(err)
		}
		gpuUS += float64(d) / float64(time.Microsecond)
	}
	b.ReportMetric(gpuUS/float64(b.N), "gpu-us/op")
}

// BenchmarkRealSmallKernelE2E measures what a tiny kernel costs end to end:
// copy a 256-byte input up, launch vector_add over 64 elements, copy the
// 256-byte result back, all on one stream. Wall time per op includes the CPU
// enqueue and the synchronize; the gpu-us/op metric is the event-timed GPU
// section, so enqueue overhead can be read apart from GPU time.
func BenchmarkRealSmallKernelE2E(b *testing.B) {
	ctx := benchRealContext(b)
	stream, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream: %v", err)
	}
	b.Cleanup(func() { _ = stream.Close() })

	const n = 64
	in, err := Alloc[float32](ctx, n)
	if err != nil {
		b.Fatalf("Alloc in: %v", err)
	}
	b.Cleanup(func() { _ = in.Close() })
	addend, err := Alloc[float32](ctx, n)
	if err != nil {
		b.Fatalf("Alloc addend: %v", err)
	}
	b.Cleanup(func() { _ = addend.Close() })
	out, err := Alloc[float32](ctx, n)
	if err != nil {
		b.Fatalf("Alloc out: %v", err)
	}
	b.Cleanup(func() { _ = out.Close() })
	hostIn, err := AllocHost[float32](ctx, n)
	if err != nil {
		b.Fatalf("AllocHost in: %v", err)
	}
	b.Cleanup(func() { _ = hostIn.Close() })
	hostOut, err := AllocHost[float32](ctx, n)
	if err != nil {
		b.Fatalf("AllocHost out: %v", err)
	}
	b.Cleanup(func() { _ = hostOut.Close() })

	mod, err := ctx.LoadModuleFromFile("testdata/vector_add.ptx")
	if err != nil {
		b.Fatalf("LoadModuleFromFile: %v", err)
	}
	b.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		b.Fatalf("Function: %v", err)
	}

	start, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent: %v", err)
	}
	b.Cleanup(func() { _ = start.Close() })
	done, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent: %v", err)
	}
	b.Cleanup(func() { _ = done.Close() })

	bg := context.Background()
	cfg := LaunchConfig1D(n, n)
	b.ResetTimer()
	var gpuUS float64
	for i := 0; i < b.N; i++ {
		_ = start.Record(stream)
		if err := in.CopyFromHostAsync(bg, stream, hostIn); err != nil {
			b.Fatal(err)
		}
		if err := fn.LaunchOn(bg, stream, cfg,
			Arg(in), Arg(addend), Arg(out), ArgValue(int32(n)),
		); err != nil {
			b.Fatal(err)
		}
		if err := out.CopyToHostAsync(bg, stream, hostOut); err != nil {
			b.Fatal(err)
		}
		_ = done.Record(stream)
		if err := stream.Synchronize(bg); err != nil {
			b.Fatal(err)
		}
		d, err := start.Elapsed(done)
		if err != nil {
			b.Fatal(err)
		}
		gpuUS += float64(d) / float64(time.Microsecond)
	}
	b.ReportMetric(gpuUS/float64(b.N), "gpu-us/op")
}
