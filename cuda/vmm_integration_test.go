//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

// TestRealVirtualMemory exercises the native VMM lifecycle on a real driver:
// allocate a VirtualBuffer, copy host data in and back, and close it.
func TestRealVirtualMemory(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	defer func() { _ = ctx.Close() }()

	const n = 4096
	vb, err := AllocVirtual[float32](ctx, n)
	if errors.Is(err, ErrSymbolUnavailable) || errors.Is(err, ErrNotSupported) {
		t.Skipf("VMM not available here: %v", err)
	}
	if err != nil {
		t.Fatalf("AllocVirtual: %v", err)
	}
	defer func() { _ = vb.Close() }()

	bg := context.Background()
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i)
	}
	if err := vb.CopyFrom(bg, in); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	out := make([]float32, n)
	if err := vb.CopyTo(bg, out); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range out {
		if out[i] != float32(i) {
			t.Fatalf("vmm[%d] = %v, want %v", i, out[i], float32(i))
		}
	}
}
