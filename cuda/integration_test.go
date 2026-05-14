//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

func initOrSkip(t *testing.T) {
	t.Helper()
	if err := Init(); err != nil {
		if errors.Is(err, ErrOperatingSystem) || errors.Is(err, ErrSystemNotReady) || errors.Is(err, ErrNoDevice) {
			t.Skipf("CUDA driver is not usable in this environment: %v", err)
		}
		t.Fatalf("Init: %v", err)
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
