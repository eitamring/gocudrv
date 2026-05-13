//go:build cuda_integration

package cuda

import (
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
