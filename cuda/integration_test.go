//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"
)

var (
	integrationInitOnce sync.Once
	integrationInitErr  error
)

// initOrSkip caches the result of Init across the integration test binary
// so a broken environment (e.g. WSL without GPU passthrough) does not
// repeatedly load and tear down the driver, which can destabilize the
// process. Tests sharing a binary share a single Init outcome.
func initOrSkip(t *testing.T) {
	t.Helper()
	integrationInitOnce.Do(func() { integrationInitErr = Init() })
	err := integrationInitErr
	if err == nil {
		if err := Init(); err != nil {
			t.Fatalf("re-Init after a unit test reset the driver: %v", err)
		}
		return
	}
	if errors.Is(err, ErrOperatingSystem) || errors.Is(err, ErrSystemNotReady) || errors.Is(err, ErrNoDevice) {
		t.Skipf("CUDA driver is not usable in this environment: %v", err)
	}
	t.Fatalf("Init: %v", err)
}

// TestRealContextCycles guards the executor teardown fix: retiring a pinned
// thread that held driver TLS used to segfault the WSL2 driver within about
// three Primary/Close cycles in one process.
func TestRealContextCycles(t *testing.T) {
	initOrSkip(t)
	for i := 0; i < 20; i++ {
		dev, err := GetDevice(0)
		if err != nil {
			t.Fatalf("cycle %d GetDevice: %v", i, err)
		}
		ctx, err := dev.Primary()
		if err != nil {
			t.Fatalf("cycle %d Primary: %v", i, err)
		}
		buf, err := Alloc[float32](ctx, 16)
		if err != nil {
			t.Fatalf("cycle %d Alloc: %v", i, err)
		}
		if err := buf.Close(); err != nil {
			t.Fatalf("cycle %d buf Close: %v", i, err)
		}
		if err := ctx.Close(); err != nil {
			t.Fatalf("cycle %d ctx Close: %v", i, err)
		}
	}
}

func TestRealInitAndVersion(t *testing.T) {
	initOrSkip(t)
	v, err := DriverVersion()
	if err != nil {
		t.Fatalf("DriverVersion: %v", err)
	}
	t.Logf("driver version: %d", v)
	if v <= 0 {
		t.Errorf("version = %d, want > 0", v)
	}
}

func TestRealPrimaryContext(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	for cycle := 0; cycle < 2; cycle++ {
		ctx, err := dev.Primary()
		if err != nil {
			t.Fatalf("Primary cycle %d: %v", cycle, err)
		}
		if err := ctx.Synchronize(context.Background()); err != nil {
			t.Errorf("Synchronize cycle %d: %v", cycle, err)
		}
		least, greatest, err := ctx.StreamPriorityRange()
		if err != nil {
			t.Errorf("StreamPriorityRange cycle %d: %v", cycle, err)
		}
		if greatest > least {
			t.Errorf("StreamPriorityRange cycle %d = (%d, %d), want greatest <= least", cycle, least, greatest)
		}
		if err := ctx.Close(); err != nil {
			t.Errorf("Close cycle %d: %v", cycle, err)
		}
		if err := ctx.Synchronize(context.Background()); !errors.Is(err, ErrContextClosed) {
			t.Errorf("Synchronize after close: err = %v, want ErrContextClosed", err)
		}
	}
}

func TestRealMemoryRoundTrip(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	const n = 1024
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 1.5
	}

	buf, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	if got := buf.Len(); got != n {
		t.Errorf("Len = %d, want %d", got, n)
	}
	if got := buf.Bytes(); got != n*4 {
		t.Errorf("Bytes = %d, want %d", got, n*4)
	}

	bg := context.Background()
	if err := buf.CopyFrom(bg, src); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	got := make([]float32, n)
	if err := buf.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("round-trip mismatch at %d: got %v, want %v", i, got[i], src[i])
		}
	}
	t.Logf("round-tripped %d float32 (%d bytes) through device memory", n, n*4)
}

func TestRealPinnedHostRoundTrip(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	const n = 1024
	hostA, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost A: %v", err)
	}
	t.Cleanup(func() { _ = hostA.Close() })
	hostB, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost B: %v", err)
	}
	t.Cleanup(func() { _ = hostB.Close() })

	srcView := hostA.Slice()
	for i := range srcView {
		srcView[i] = float32(i) * 0.25
	}

	dev0, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc device: %v", err)
	}
	t.Cleanup(func() { _ = dev0.Close() })

	bg := context.Background()
	if err := dev0.CopyFromHost(bg, hostA); err != nil {
		t.Fatalf("CopyFromHost: %v", err)
	}
	if err := dev0.CopyToHost(bg, hostB); err != nil {
		t.Fatalf("CopyToHost: %v", err)
	}

	a := hostA.Slice()
	b := hostB.Slice()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("round-trip mismatch at %d: a=%v b=%v", i, a[i], b[i])
		}
	}
	t.Logf("round-tripped %d float32 (%d bytes) through pinned host and device buffers", n, n*4)
}

func TestRealPinnedHostAsyncRoundTrip(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	const n = 1024
	hostA, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost A: %v", err)
	}
	t.Cleanup(func() { _ = hostA.Close() })
	hostB, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost B: %v", err)
	}
	t.Cleanup(func() { _ = hostB.Close() })
	dev0, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc device: %v", err)
	}
	t.Cleanup(func() { _ = dev0.Close() })

	srcView := hostA.Slice()
	for i := range srcView {
		srcView[i] = float32(i) * 0.5
	}

	bg := context.Background()
	if err := dev0.CopyFromHostAsync(bg, stream, hostA); err != nil {
		t.Fatalf("CopyFromHostAsync: %v", err)
	}
	if err := dev0.CopyToHostAsync(bg, stream, hostB); err != nil {
		t.Fatalf("CopyToHostAsync: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Stream.Synchronize: %v", err)
	}

	a := hostA.Slice()
	b := hostB.Slice()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("async round-trip mismatch at %d: a=%v b=%v", i, a[i], b[i])
		}
	}
	t.Logf("async round-tripped %d float32 (%d bytes) through pinned host and device buffers", n, n*4)
}

func TestRealEventStreamWait(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	copyIn, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream copyIn: %v", err)
	}
	t.Cleanup(func() { _ = copyIn.Close() })
	copyOut, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream copyOut: %v", err)
	}
	t.Cleanup(func() { _ = copyOut.Close() })
	start, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent start: %v", err)
	}
	t.Cleanup(func() { _ = start.Close() })
	ready, err := ctx.NewEvent(WithEventDisableTiming())
	if err != nil {
		t.Fatalf("NewEvent ready: %v", err)
	}
	t.Cleanup(func() { _ = ready.Close() })
	done, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent done: %v", err)
	}
	t.Cleanup(func() { _ = done.Close() })

	const n = 1024
	hostA, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost A: %v", err)
	}
	t.Cleanup(func() { _ = hostA.Close() })
	hostB, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost B: %v", err)
	}
	t.Cleanup(func() { _ = hostB.Close() })
	dev0, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc device: %v", err)
	}
	t.Cleanup(func() { _ = dev0.Close() })

	srcView := hostA.Slice()
	for i := range srcView {
		srcView[i] = float32(i) * 0.75
	}

	bg := context.Background()
	if err := start.Record(copyIn); err != nil {
		t.Fatalf("start.Record: %v", err)
	}
	if err := dev0.CopyFromHostAsync(bg, copyIn, hostA); err != nil {
		t.Fatalf("CopyFromHostAsync: %v", err)
	}
	if err := ready.Record(copyIn); err != nil {
		t.Fatalf("ready.Record: %v", err)
	}
	if err := ready.Query(); !errors.Is(err, ErrNotReady) && err != nil {
		t.Fatalf("ready.Query: %v", err)
	}
	if err := copyOut.WaitEvent(ready); err != nil {
		t.Fatalf("WaitEvent: %v", err)
	}
	if err := dev0.CopyToHostAsync(bg, copyOut, hostB); err != nil {
		t.Fatalf("CopyToHostAsync: %v", err)
	}
	if err := done.Record(copyOut); err != nil {
		t.Fatalf("done.Record: %v", err)
	}
	if err := done.Synchronize(bg); err != nil {
		t.Fatalf("done.Synchronize: %v", err)
	}
	elapsed, err := start.Elapsed(done)
	if err != nil {
		t.Fatalf("Elapsed: %v", err)
	}

	a := hostA.Slice()
	b := hostB.Slice()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("event-ordered round-trip mismatch at %d: a=%v b=%v", i, a[i], b[i])
		}
	}
	t.Logf("event-ordered async round-trip %d float32 (%d bytes), elapsed %v", n, n*4, elapsed)
}

func TestRealMemoryPrimitives(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	free, total, err := ctx.MemInfo()
	if err != nil {
		t.Fatalf("MemInfo: %v", err)
	}
	if total == 0 || free == 0 || free > total {
		t.Fatalf("MemInfo returned implausible values: free=%d total=%d", free, total)
	}

	const n = 1024
	src, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc src: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })
	dst, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })
	host, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = host.Close() })

	bg := context.Background()
	if err := src.Zero(bg); err != nil {
		t.Fatalf("Zero: %v", err)
	}
	if err := src.CopyToHost(bg, host); err != nil {
		t.Fatalf("CopyToHost after Zero: %v", err)
	}
	for i, v := range host.Slice() {
		if v != 0 {
			t.Fatalf("Zero left nonzero at %d: %v", i, v)
		}
	}

	const fillVal = float32(3.5)
	if err := src.Fill(bg, fillVal); err != nil {
		t.Fatalf("Fill: %v", err)
	}
	if err := src.CopyToHost(bg, host); err != nil {
		t.Fatalf("CopyToHost after Fill: %v", err)
	}
	for i, v := range host.Slice() {
		if v != fillVal {
			t.Fatalf("Fill wrote %v at %d, want %v", v, i, fillVal)
		}
	}

	wide, err := Alloc[float64](ctx, n)
	if err != nil {
		t.Fatalf("Alloc wide: %v", err)
	}
	t.Cleanup(func() { _ = wide.Close() })
	if err := wide.Fill(bg, 1); !errors.Is(err, ErrUnsupportedFillType) {
		t.Fatalf("Fill on float64 = %v, want ErrUnsupportedFillType", err)
	}

	feed := host.Slice()
	for i := range feed {
		feed[i] = float32(i) + 1
	}
	if err := src.CopyFromHost(bg, host); err != nil {
		t.Fatalf("CopyFromHost: %v", err)
	}
	if err := src.CopyToDevice(bg, dst); err != nil {
		t.Fatalf("CopyToDevice: %v", err)
	}
	readback, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost readback: %v", err)
	}
	t.Cleanup(func() { _ = readback.Close() })
	if err := dst.CopyToHost(bg, readback); err != nil {
		t.Fatalf("CopyToHost from dst: %v", err)
	}
	got := readback.Slice()
	for i := range got {
		if got[i] != float32(i)+1 {
			t.Fatalf("device-to-device mismatch at %d: got %v want %v", i, got[i], float32(i)+1)
		}
	}
	t.Logf("primitives ok: free=%d total=%d, zeroed and DtoD-copied %d float32", free, total, n)
}

func TestRealModuleLoad(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	ptx, err := os.ReadFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("read ptx: %v", err)
	}

	mod, err := ctx.LoadModule(ptx)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	if fn.Name() != "vector_add" {
		t.Errorf("Name = %q, want vector_add", fn.Name())
	}
	t.Logf("loaded module with vector_add function")
}

func TestRealVectorAddLaunch(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	const n = 1024
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	for i := range aHost {
		aHost[i] = float32(i)
		bHost[i] = float32(i) * 2
	}

	a, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	out, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	t.Cleanup(func() { _ = out.Close() })

	bg := context.Background()
	if err := a.CopyFrom(bg, aHost); err != nil {
		t.Fatalf("CopyFrom a: %v", err)
	}
	if err := b.CopyFrom(bg, bHost); err != nil {
		t.Fatalf("CopyFrom b: %v", err)
	}

	mod, err := ctx.LoadModuleFromFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	if err := fn.Launch(bg, LaunchConfig1D(n, 256),
		Arg(a),
		Arg(b),
		Arg(out),
		ArgValue(int32(n)),
	); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, n)
	if err := out.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo out: %v", err)
	}
	for i := range got {
		want := aHost[i] + bHost[i]
		if got[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, got[i], want)
		}
	}
	t.Logf("launched vector_add for %d elements", n)
}

func TestRealVectorAddLaunchOnStream(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	const n = 1024
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	for i := range aHost {
		aHost[i] = float32(i)
		bHost[i] = float32(i) * 2
	}

	a, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	out, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	t.Cleanup(func() { _ = out.Close() })

	bg := context.Background()
	if err := a.CopyFrom(bg, aHost); err != nil {
		t.Fatalf("CopyFrom a: %v", err)
	}
	if err := b.CopyFrom(bg, bHost); err != nil {
		t.Fatalf("CopyFrom b: %v", err)
	}

	mod, err := ctx.LoadModuleFromFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	if err := fn.LaunchOn(bg, stream, LaunchConfig1D(n, 256),
		Arg(a),
		Arg(b),
		Arg(out),
		ArgValue(int32(n)),
	); err != nil {
		t.Fatalf("LaunchOn: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Stream.Synchronize: %v", err)
	}

	got := make([]float32, n)
	if err := out.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo out: %v", err)
	}
	for i := range got {
		want := aHost[i] + bHost[i]
		if got[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, got[i], want)
		}
	}
	t.Logf("launched vector_add for %d elements on an explicit stream", n)
}

func TestRealDeviceEnum(t *testing.T) {
	initOrSkip(t)
	n, err := DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	t.Logf("devices: %d", n)
	if n <= 0 {
		t.Skip("no CUDA devices available")
	}
	for i := 0; i < n; i++ {
		d, err := GetDevice(i)
		if err != nil {
			t.Fatalf("GetDevice(%d): %v", i, err)
		}
		name, err := d.Name()
		if err != nil {
			t.Errorf("Name: %v", err)
		}
		mem, err := d.TotalMemory()
		if err != nil {
			t.Errorf("TotalMemory: %v", err)
		}
		maj, min, err := d.ComputeCapability()
		if err != nil {
			t.Errorf("ComputeCapability: %v", err)
		}
		sm, err := d.Attribute(DeviceAttributeMultiprocessorCount)
		if err != nil {
			t.Errorf("Attribute: %v", err)
		}
		t.Logf("device %d: %q, cc %d.%d, memory %d MiB, %d SMs",
			i, name, maj, min, mem/(1<<20), sm)
	}
}

func TestRealOccupancy(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	mod, err := ctx.LoadModuleFromFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	blocks, err := fn.MaxActiveBlocksPerSM(256, 0)
	if err != nil {
		t.Fatalf("MaxActiveBlocksPerSM: %v", err)
	}
	if blocks <= 0 {
		t.Fatalf("MaxActiveBlocksPerSM = %d, want positive", blocks)
	}

	minGrid, blockSize, err := fn.SuggestedBlockSize(0, 0)
	if err != nil {
		t.Fatalf("SuggestedBlockSize: %v", err)
	}
	if blockSize <= 0 || minGrid <= 0 {
		t.Fatalf("SuggestedBlockSize = (minGrid %d, block %d), want both positive", minGrid, blockSize)
	}

	cfg, err := fn.SuggestedConfig1D(1<<20, 0)
	if err != nil {
		t.Fatalf("SuggestedConfig1D: %v", err)
	}
	if cfg.BlockX == 0 || cfg.GridX == 0 {
		t.Fatalf("SuggestedConfig1D produced empty config: %+v", cfg)
	}
	t.Logf("occupancy ok: %d blocks/SM at 256, suggested block %d (min grid %d), 1M config %dx%d",
		blocks, blockSize, minGrid, cfg.GridX, cfg.BlockX)
}

func TestRealDeviceGlobal(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	mod, err := ctx.LoadModuleFromFile("testdata/globals.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })

	g, err := mod.Global("g_counter")
	if err != nil {
		t.Fatalf("Global: %v", err)
	}
	if g.Bytes() != 16 {
		t.Fatalf("Bytes = %d, want 16 (4 uint32)", g.Bytes())
	}

	bg := context.Background()
	want := []uint32{11, 22, 33, 44}
	if err := WriteGlobal(bg, g, want); err != nil {
		t.Fatalf("WriteGlobal: %v", err)
	}
	got := make([]uint32, 4)
	if err := ReadGlobal(bg, got, g); err != nil {
		t.Fatalf("ReadGlobal: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("global round trip mismatch at %d: got %d want %d", i, got[i], want[i])
		}
	}
	t.Logf("device global ok: g_counter (%d bytes) round-tripped %v", g.Bytes(), got)
}

func TestRealGraphCaptureReplay(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	const n = 1024
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	for i := range aHost {
		aHost[i] = float32(i)
		bHost[i] = float32(i) * 2
	}
	a, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	out, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	t.Cleanup(func() { _ = out.Close() })

	bg := context.Background()
	if err := a.CopyFrom(bg, aHost); err != nil {
		t.Fatalf("CopyFrom a: %v", err)
	}
	if err := b.CopyFrom(bg, bHost); err != nil {
		t.Fatalf("CopyFrom b: %v", err)
	}

	mod, err := ctx.LoadModuleFromFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	// Capture a launch into a graph instead of running it.
	if err := stream.BeginCapture(CaptureModeThreadLocal); err != nil {
		t.Fatalf("BeginCapture: %v", err)
	}
	if err := fn.LaunchOn(bg, stream, LaunchConfig1D(n, 256), Arg(a), Arg(b), Arg(out), ArgValue(int32(n))); err != nil {
		t.Fatalf("LaunchOn during capture: %v", err)
	}
	g, err := stream.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate: %v", err)
	}
	t.Cleanup(func() { _ = exec.Close() })

	// Replay the graph and check the result lands.
	if err := exec.Launch(bg, stream); err != nil {
		t.Fatalf("graph Launch: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got := make([]float32, n)
	if err := out.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo out: %v", err)
	}
	for i := range got {
		if want := aHost[i] + bHost[i]; got[i] != want {
			t.Fatalf("graph replay out[%d] = %v, want %v", i, got[i], want)
		}
	}
	t.Logf("graph capture/replay ok: vector_add for %d elements replayed from a graph", n)
}

func TestRealAsyncAllocRoundTrip(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	const n = 1024
	buf, err := AllocAsync[float32](ctx, stream, n)
	if err != nil {
		t.Fatalf("AllocAsync: %v", err)
	}

	bg := context.Background()
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 1.5
	}
	if err := buf.CopyFrom(bg, src); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	got := make([]float32, n)
	if err := buf.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range got {
		if got[i] != src[i] {
			t.Fatalf("async-alloc round-trip mismatch at %d: got=%v want=%v", i, got[i], src[i])
		}
	}

	// Free on the stream, then make sure the work has drained before the
	// context is torn down.
	if err := buf.FreeAsync(stream); err != nil {
		t.Fatalf("FreeAsync: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	t.Logf("async alloc/free ok: round-tripped %d float32 through stream-ordered memory", n)
}

func TestRealMemoryPool(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	pool, err := ctx.DefaultMemPool()
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("memory pools not supported on this driver")
	}
	if err != nil {
		t.Fatalf("DefaultMemPool: %v", err)
	}
	if err := pool.SetReleaseThreshold(1 << 20); err != nil {
		t.Fatalf("SetReleaseThreshold: %v", err)
	}
	if got, err := pool.ReleaseThreshold(); err != nil || got != 1<<20 {
		t.Fatalf("ReleaseThreshold = %d, %v; want 1048576", got, err)
	}

	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	const n = 4096
	buf, err := AllocFromPool[float32](pool, stream, n)
	if err != nil {
		t.Fatalf("AllocFromPool: %v", err)
	}
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)
	}
	if err := buf.CopyFromHostAsync(context.Background(), stream, mustHost(t, ctx, src)); err != nil {
		t.Fatalf("CopyFromHostAsync: %v", err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if err := buf.FreeAsync(stream); err != nil {
		t.Fatalf("FreeAsync: %v", err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize after free: %v", err)
	}
}

func mustHost(t *testing.T, ctx *Context, src []float32) *HostBuffer[float32] {
	t.Helper()
	h, err := AllocHost[float32](ctx, len(src))
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })
	copy(h.Slice(), src)
	return h
}

func TestRealModuleJITLog(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	// Deliberately malformed PTX so the JIT compile fails and writes a log.
	bad := []byte(".version 7.0\n.target sm_50\n.address_size 64\nthis is not valid ptx\n")
	mod, log, err := ctx.LoadModuleEx(bad, JITOptions{})
	if err == nil {
		_ = mod.Close()
		t.Fatal("expected a JIT compile error for malformed PTX")
	}
	if log.Error == "" {
		t.Error("expected a non-empty JIT error log on compile failure")
	} else {
		t.Logf("JIT error log: %s", log.Error)
	}
}

func TestRealLinkerVectorAdd(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	ptx, err := os.ReadFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("read ptx: %v", err)
	}

	lk, err := ctx.NewLinker(JITOptions{})
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("JIT linker not supported on this driver")
	}
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	t.Cleanup(func() { _ = lk.Close() })

	if err := lk.AddPTX("vector_add.ptx", ptx); err != nil {
		t.Fatalf("AddPTX: %v (log: %s)", err, lk.Log().Error)
	}
	image, err := lk.Complete()
	if err != nil {
		t.Fatalf("Complete: %v (log: %s)", err, lk.Log().Error)
	}
	if len(image) == 0 {
		t.Fatal("linked image is empty")
	}

	mod, err := ctx.LoadModule(image)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	const n = 64
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	for i := range aHost {
		aHost[i] = float32(i)
		bHost[i] = float32(i) * 2
	}
	a, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	out, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	t.Cleanup(func() { _ = out.Close() })

	bg := context.Background()
	if err := a.CopyFrom(bg, aHost); err != nil {
		t.Fatalf("CopyFrom a: %v", err)
	}
	if err := b.CopyFrom(bg, bHost); err != nil {
		t.Fatalf("CopyFrom b: %v", err)
	}
	if err := fn.Launch(bg, LaunchConfig1D(n, 64), Arg(a), Arg(b), Arg(out), ArgValue(int32(n))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got := make([]float32, n)
	if err := out.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo out: %v", err)
	}
	for i := range got {
		if want := aHost[i] + bHost[i]; got[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, got[i], want)
		}
	}
	t.Logf("linked vector_add from PTX and launched for %d elements", n)
}

// TestRealLinkerBadPTX feeds the linker malformed PTX, which the driver rejects
// at AddData or, if it defers, at Complete; either way the error log must fill.
func TestRealLinkerBadPTX(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	lk, err := ctx.NewLinker(JITOptions{})
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("JIT linker not supported on this driver")
	}
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	t.Cleanup(func() { _ = lk.Close() })

	err = lk.AddPTX("bad.ptx", []byte(".version 99.9\ngarbage"))
	if err == nil {
		_, err = lk.Complete()
	}
	if err == nil {
		t.Fatal("expected a link error for malformed PTX")
	}
	if lk.Log().Error == "" {
		t.Error("expected a non-empty error log on link failure")
	} else {
		t.Logf("JIT link error log: %s", lk.Log().Error)
	}
}

func TestRealLaunchPackedSet(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })

	const n = 8
	const sentinel = float32(-1)
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	out := make([]float32, n)
	for i := range aHost {
		aHost[i] = 1
		bHost[i] = 2
		out[i] = sentinel
	}
	bufA, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	t.Cleanup(func() { _ = bufA.Close() })
	bufB, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc b: %v", err)
	}
	t.Cleanup(func() { _ = bufB.Close() })
	bufOut, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	t.Cleanup(func() { _ = bufOut.Close() })

	bg := context.Background()
	if err := bufA.CopyFrom(bg, aHost); err != nil {
		t.Fatalf("CopyFrom a: %v", err)
	}
	if err := bufB.CopyFrom(bg, bHost); err != nil {
		t.Fatalf("CopyFrom b: %v", err)
	}
	if err := bufOut.CopyFrom(bg, out); err != nil {
		t.Fatalf("CopyFrom out: %v", err)
	}

	mod, err := ctx.LoadModuleFromFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("vector_add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	p, err := Pack(Arg(bufA), Arg(bufB), Arg(bufOut), ArgValue(int32(4)))
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	cfg := LaunchConfig1D(n, n)
	if err := fn.LaunchPacked(bg, cfg, p); err != nil {
		t.Fatalf("LaunchPacked n=4: %v", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if err := bufOut.CopyTo(bg, out); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := 0; i < 4; i++ {
		if out[i] != 3 {
			t.Errorf("out[%d] = %v, want 3 after n=4 launch", i, out[i])
		}
	}
	for i := 4; i < n; i++ {
		if out[i] != sentinel {
			t.Errorf("out[%d] = %v, want sentinel after n=4 launch", i, out[i])
		}
	}

	if err := SetPacked(p, 3, int32(n)); err != nil {
		t.Fatalf("SetPacked: %v", err)
	}
	if err := fn.LaunchPacked(bg, cfg, p); err != nil {
		t.Fatalf("LaunchPacked n=8: %v", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize 2: %v", err)
	}
	if err := bufOut.CopyTo(bg, out); err != nil {
		t.Fatalf("CopyTo 2: %v", err)
	}
	for i := range out {
		if out[i] != 3 {
			t.Errorf("out[%d] = %v, want 3 after n=8 launch", i, out[i])
		}
	}
}
