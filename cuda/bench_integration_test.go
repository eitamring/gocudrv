//go:build cuda_integration

package cuda

import (
	"context"
	"testing"
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
		gpuUS += float64(d.Microseconds())
	}
	b.ReportMetric(gpuUS/float64(b.N), "gpu-us/op")
}
