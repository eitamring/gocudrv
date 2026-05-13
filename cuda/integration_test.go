//go:build cuda_integration

package cuda

import "testing"

func TestRealInitAndVersion(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init: %v", err)
	}
	v, err := DriverVersion()
	if err != nil {
		t.Fatalf("DriverVersion: %v", err)
	}
	t.Logf("driver version: %d", v)
	if v <= 0 {
		t.Errorf("version = %d, want > 0", v)
	}
}
