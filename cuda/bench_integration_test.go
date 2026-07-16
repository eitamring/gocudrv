//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"sync"
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

// BenchmarkRealTwoStreamWait parks one wait lane inside the driver on stream A
// via a blocking host function, then measures repeated stream B synchronizes to
// confirm an unrelated wait does not queue behind the parked lane.
func BenchmarkRealTwoStreamWait(b *testing.B) {
	ctx := benchRealContext(b)
	streamA, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream A: %v", err)
	}
	b.Cleanup(func() { _ = streamA.Close() })
	streamB, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream B: %v", err)
	}
	b.Cleanup(func() { _ = streamB.Close() })

	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	b.Cleanup(unblock)
	err = streamA.LaunchHostFunc(func() {
		close(entered)
		<-release
	})
	if errors.Is(err, ErrSymbolUnavailable) {
		b.Skip("driver lacks cuLaunchHostFunc")
	}
	if err != nil {
		b.Fatalf("LaunchHostFunc: %v", err)
	}
	syncA := make(chan error, 1)
	go func() { syncA <- streamA.Synchronize(context.Background()) }()
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		b.Fatal("host function did not start")
	}
	waitForActiveSyncs(b, ctx, 1)

	bg := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := streamB.Synchronize(bg); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	unblock()
	if err := <-syncA; err != nil {
		b.Fatalf("Synchronize A: %v", err)
	}
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

const benchThroughputBytes = 64 << 20

func reportRealGPUMetrics(b *testing.B, bytesPerOp int64, totalUS float64) {
	if b.N == 0 || totalUS <= 0 {
		return
	}
	b.ReportMetric(totalUS/float64(b.N), "gpu-us/op")
	b.ReportMetric(float64(bytesPerOp)*float64(b.N)/(totalUS*1e3), "gpu-GB/s")
}

// BenchmarkRealCopyOverlap compares two opposite-direction copies serialized
// on one stream with the same copies running on separate streams. Events on a
// control stream measure the interval until both transfer streams finish.
func BenchmarkRealCopyOverlap(b *testing.B) {
	ctx := benchRealContext(b)
	control, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream control: %v", err)
	}
	b.Cleanup(func() { _ = control.Close() })
	upload, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream upload: %v", err)
	}
	b.Cleanup(func() { _ = upload.Close() })
	download, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream download: %v", err)
	}
	b.Cleanup(func() { _ = download.Close() })

	uploadDevice, err := Alloc[uint8](ctx, benchThroughputBytes)
	if err != nil {
		b.Fatalf("Alloc upload: %v", err)
	}
	b.Cleanup(func() { _ = uploadDevice.Close() })
	downloadDevice, err := Alloc[uint8](ctx, benchThroughputBytes)
	if err != nil {
		b.Fatalf("Alloc download: %v", err)
	}
	b.Cleanup(func() { _ = downloadDevice.Close() })
	uploadHost, err := AllocHost[uint8](ctx, benchThroughputBytes)
	if err != nil {
		b.Fatalf("AllocHost upload: %v", err)
	}
	b.Cleanup(func() { _ = uploadHost.Close() })
	downloadHost, err := AllocHost[uint8](ctx, benchThroughputBytes)
	if err != nil {
		b.Fatalf("AllocHost download: %v", err)
	}
	b.Cleanup(func() { _ = downloadHost.Close() })

	start, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent start: %v", err)
	}
	b.Cleanup(func() { _ = start.Close() })
	uploadDone, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent upload: %v", err)
	}
	b.Cleanup(func() { _ = uploadDone.Close() })
	downloadDone, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent download: %v", err)
	}
	b.Cleanup(func() { _ = downloadDone.Close() })
	done, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent done: %v", err)
	}
	b.Cleanup(func() { _ = done.Close() })

	bg := context.Background()
	bytesPerOp := int64(2 * benchThroughputBytes)
	b.Run("serial", func(b *testing.B) {
		b.SetBytes(bytesPerOp)
		b.ResetTimer()
		var gpuUS float64
		for i := 0; i < b.N; i++ {
			if err := start.Record(control); err != nil {
				b.Fatal(err)
			}
			if err := uploadDevice.CopyFromHostAsync(bg, control, uploadHost); err != nil {
				b.Fatal(err)
			}
			if err := downloadDevice.CopyToHostAsync(bg, control, downloadHost); err != nil {
				b.Fatal(err)
			}
			if err := done.Record(control); err != nil {
				b.Fatal(err)
			}
			if err := control.Synchronize(bg); err != nil {
				b.Fatal(err)
			}
			d, err := start.Elapsed(done)
			if err != nil {
				b.Fatal(err)
			}
			gpuUS += float64(d) / float64(time.Microsecond)
		}
		reportRealGPUMetrics(b, bytesPerOp, gpuUS)
	})
	b.Run("parallel", func(b *testing.B) {
		b.SetBytes(bytesPerOp)
		b.ResetTimer()
		var gpuUS float64
		for i := 0; i < b.N; i++ {
			if err := start.Record(control); err != nil {
				b.Fatal(err)
			}
			if err := upload.WaitEvent(start); err != nil {
				b.Fatal(err)
			}
			if err := download.WaitEvent(start); err != nil {
				b.Fatal(err)
			}
			if err := uploadDevice.CopyFromHostAsync(bg, upload, uploadHost); err != nil {
				b.Fatal(err)
			}
			if err := downloadDevice.CopyToHostAsync(bg, download, downloadHost); err != nil {
				b.Fatal(err)
			}
			if err := uploadDone.Record(upload); err != nil {
				b.Fatal(err)
			}
			if err := downloadDone.Record(download); err != nil {
				b.Fatal(err)
			}
			if err := control.WaitEvent(uploadDone); err != nil {
				b.Fatal(err)
			}
			if err := control.WaitEvent(downloadDone); err != nil {
				b.Fatal(err)
			}
			if err := done.Record(control); err != nil {
				b.Fatal(err)
			}
			if err := control.Synchronize(bg); err != nil {
				b.Fatal(err)
			}
			d, err := start.Elapsed(done)
			if err != nil {
				b.Fatal(err)
			}
			gpuUS += float64(d) / float64(time.Microsecond)
		}
		reportRealGPUMetrics(b, bytesPerOp, gpuUS)
	})
}

// BenchmarkRealMemsetThroughput measures an event-timed asynchronous device
// memset over a 64 MiB byte buffer.
func BenchmarkRealMemsetThroughput(b *testing.B) {
	ctx := benchRealContext(b)
	stream, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream: %v", err)
	}
	b.Cleanup(func() { _ = stream.Close() })
	buf, err := Alloc[uint8](ctx, benchThroughputBytes)
	if err != nil {
		b.Fatalf("Alloc: %v", err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	start, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent start: %v", err)
	}
	b.Cleanup(func() { _ = start.Close() })
	done, err := ctx.NewEvent()
	if err != nil {
		b.Fatalf("NewEvent done: %v", err)
	}
	b.Cleanup(func() { _ = done.Close() })

	bg := context.Background()
	b.SetBytes(benchThroughputBytes)
	b.ResetTimer()
	var gpuUS float64
	for i := 0; i < b.N; i++ {
		if err := start.Record(stream); err != nil {
			b.Fatal(err)
		}
		if err := buf.ZeroAsync(bg, stream); err != nil {
			b.Fatal(err)
		}
		if err := done.Record(stream); err != nil {
			b.Fatal(err)
		}
		if err := stream.Synchronize(bg); err != nil {
			b.Fatal(err)
		}
		d, err := start.Elapsed(done)
		if err != nil {
			b.Fatal(err)
		}
		gpuUS += float64(d) / float64(time.Microsecond)
	}
	reportRealGPUMetrics(b, benchThroughputBytes, gpuUS)
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
