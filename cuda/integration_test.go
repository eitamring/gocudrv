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
		return
	}
	if errors.Is(err, ErrOperatingSystem) || errors.Is(err, ErrSystemNotReady) || errors.Is(err, ErrNoDevice) {
		t.Skipf("CUDA driver is not usable in this environment: %v", err)
	}
	t.Fatalf("Init: %v", err)
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
