package cuda

import (
	"context"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// benchDriver is a fake driver whose calls return immediately, so these
// benchmarks measure the gocudrv wrapper and executor overhead (the CPU enqueue
// cost), not real GPU time. The GPU-bound benchmarks live behind the
// cuda_integration tag.
func benchDriver() *cudasys.Driver {
	d := fakeMemoryDriver(&memCalls{}, 0xB000)
	d.CuMemAllocHost = func(pp **byte, n uint64) cudasys.CUresult {
		s := make([]byte, n)
		*pp = &s[0]
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemFreeHost = func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuStreamCreate = func(s *cudasys.CUstream, _ uint32) cudasys.CUresult { *s = 0x5151; return cudasys.CUDA_SUCCESS }
	d.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuStreamSynchronize = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuEventCreate = func(e *cudasys.CUevent, _ uint32) cudasys.CUresult { *e = 0xE0E0; return cudasys.CUDA_SUCCESS }
	d.CuEventDestroy = func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuEventRecord = func(cudasys.CUevent, cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuMemAllocPitch = func(ptr *cudasys.CUdeviceptr, pitch *uint64, _, _ uint64, _ uint32) cudasys.CUresult {
		*ptr = 0xB000
		*pitch = 512
		return cudasys.CUDA_SUCCESS
	}
	d.CuMemcpy2D = func(*cudasys.Memcpy2D) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuStreamBeginCapture = func(cudasys.CUstream, uint32) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuStreamEndCapture = func(_ cudasys.CUstream, g *cudasys.CUgraph) cudasys.CUresult {
		*g = 0x6A6A
		return cudasys.CUDA_SUCCESS
	}
	d.CuGraphInstantiate = func(e *cudasys.CUgraphExec, _ cudasys.CUgraph, _ uint64) cudasys.CUresult {
		*e = 0x7E7E
		return cudasys.CUDA_SUCCESS
	}
	d.CuGraphLaunch = func(cudasys.CUgraphExec, cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuGraphDestroy = func(cudasys.CUgraph) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuGraphExecDestroy = func(cudasys.CUgraphExec) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuLaunchKernel = func(cudasys.CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, cudasys.CUstream, *unsafe.Pointer, *unsafe.Pointer) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	d.CuModuleLoadData = func(m *cudasys.CUmodule, _ *byte) cudasys.CUresult { *m = 0xBEEF; return cudasys.CUDA_SUCCESS }
	d.CuModuleUnload = func(cudasys.CUmodule) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	d.CuModuleGetFunction = func(f *cudasys.CUfunction, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
		*f = 0xCAFE
		return cudasys.CUDA_SUCCESS
	}
	return d
}

func benchContext(b *testing.B) *Context {
	b.Helper()
	resetDriver()
	mu.Lock()
	driver = benchDriver()
	mu.Unlock()
	b.Cleanup(resetDriver)
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

func BenchmarkPrimaryContext(b *testing.B) {
	resetDriver()
	mu.Lock()
	driver = benchDriver()
	mu.Unlock()
	b.Cleanup(resetDriver)
	dev, err := GetDevice(0)
	if err != nil {
		b.Fatalf("GetDevice: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, err := dev.Primary()
		if err != nil {
			b.Fatal(err)
		}
		_ = ctx.Close()
	}
}

func BenchmarkAllocFree(b *testing.B) {
	ctx := benchContext(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, err := Alloc[float32](ctx, 1024)
		if err != nil {
			b.Fatal(err)
		}
		_ = buf.Close()
	}
}

func BenchmarkAsyncAllocFree(b *testing.B) {
	ctx := benchContext(b)
	stream, _ := ctx.NewStream()
	b.Cleanup(func() { _ = stream.Close() })
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, err := AllocAsync[float32](ctx, stream, 1024)
		if err != nil {
			b.Fatal(err)
		}
		_ = buf.FreeAsync(stream)
	}
}

func BenchmarkPageableCopy(b *testing.B) {
	ctx := benchContext(b)
	const n = 4096
	buf, _ := Alloc[float32](ctx, n)
	b.Cleanup(func() { _ = buf.Close() })
	src := make([]float32, n)
	b.SetBytes(n * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyFrom(context.Background(), src); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPinnedCopy(b *testing.B) {
	ctx := benchContext(b)
	const n = 4096
	buf, _ := Alloc[float32](ctx, n)
	b.Cleanup(func() { _ = buf.Close() })
	host, _ := AllocHost[float32](ctx, n)
	b.Cleanup(func() { _ = host.Close() })
	b.SetBytes(n * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyFromHost(context.Background(), host); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAsyncPinnedCopy(b *testing.B) {
	ctx := benchContext(b)
	stream, _ := ctx.NewStream()
	b.Cleanup(func() { _ = stream.Close() })
	const n = 4096
	buf, _ := Alloc[float32](ctx, n)
	b.Cleanup(func() { _ = buf.Close() })
	host, _ := AllocHost[float32](ctx, n)
	b.Cleanup(func() { _ = host.Close() })
	b.SetBytes(n * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyFromHostAsync(context.Background(), stream, host); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPageableCopyDtoH(b *testing.B) {
	ctx := benchContext(b)
	const n = 4096
	buf, _ := Alloc[float32](ctx, n)
	b.Cleanup(func() { _ = buf.Close() })
	dst := make([]float32, n)
	b.SetBytes(n * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyTo(context.Background(), dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFill(b *testing.B) {
	ctx := benchContext(b)
	const n = 4096
	buf, _ := Alloc[float32](ctx, n)
	b.Cleanup(func() { _ = buf.Close() })
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.Fill(context.Background(), 1.5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGraphLaunch(b *testing.B) {
	ctx := benchContext(b)
	stream, _ := ctx.NewStream()
	b.Cleanup(func() { _ = stream.Close() })
	if err := stream.BeginCapture(CaptureModeThreadLocal); err != nil {
		b.Fatalf("BeginCapture: %v", err)
	}
	g, _ := stream.EndCapture()
	b.Cleanup(func() { _ = g.Close() })
	exec, _ := g.Instantiate()
	b.Cleanup(func() { _ = exec.Close() })
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := exec.Launch(context.Background(), stream); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStreamSynchronize(b *testing.B) {
	ctx := benchContext(b)
	stream, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream: %v", err)
	}
	b.Cleanup(func() { _ = stream.Close() })
	bg := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := stream.Synchronize(bg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEventRecord(b *testing.B) {
	ctx := benchContext(b)
	stream, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream: %v", err)
	}
	b.Cleanup(func() { _ = stream.Close() })
	ev, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent: %v", err)
	}
	b.Cleanup(func() { _ = ev.Close() })
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ev.Record(stream); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPitchedCopy(b *testing.B) {
	ctx := benchContext(b)
	pb, err := AllocPitched[float32](ctx, 64, 8)
	if err != nil {
		b.Fatalf("AllocPitched: %v", err)
	}
	b.Cleanup(func() { _ = pb.Close() })
	src := make([]float32, 64*8)
	bg := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := pb.CopyFrom(bg, src); err != nil {
			b.Fatal(err)
		}
	}
}
