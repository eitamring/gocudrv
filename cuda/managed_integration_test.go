//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

// TestRealManagedMemory exercises the native cuMemAllocManaged / prefetch /
// advise path on a real driver: it writes the host Slice, migrates the pages to
// the device and back, and confirms the data survives. Prefetch and advise are
// tolerated as not-supported on devices without concurrent managed access.
func TestRealManagedMemory(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	defer ctx.Close()

	const n = 4096
	mb, err := AllocManaged[float32](ctx, n)
	if errors.Is(err, ErrSymbolUnavailable) || errors.Is(err, ErrNotSupported) {
		t.Skipf("managed memory not available here: %v", err)
	}
	if err != nil {
		t.Fatalf("AllocManaged: %v", err)
	}
	defer func() { _ = mb.Close() }()

	s := mb.Slice()
	if len(s) != n {
		t.Fatalf("Slice len = %d, want %d", len(s), n)
	}
	for i := range s {
		s[i] = float32(i)
	}

	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	bg := context.Background()

	if err := mb.Advise(AdviseSetReadMostly); err != nil && !errors.Is(err, ErrNotSupported) {
		t.Errorf("Advise: %v", err)
	}
	cma, err := dev.Attribute(DeviceAttributeConcurrentManagedAccess)
	if err != nil {
		t.Fatalf("concurrent-managed-access attribute: %v", err)
	}
	if cma == 0 {
		t.Logf("no concurrent managed access (e.g. WSL2); skipping prefetch")
	} else {
		for _, p := range []struct {
			name string
			fn   func(context.Context, *Stream) error
		}{{"to-device", mb.PrefetchToDevice}, {"to-host", mb.PrefetchToHost}} {
			if err := p.fn(bg, stream); err != nil {
				if errors.Is(err, ErrNotSupported) {
					t.Logf("prefetch %s not supported on this device", p.name)
					continue
				}
				t.Errorf("Prefetch %s: %v", p.name, err)
			}
			if err := stream.Synchronize(bg); err != nil {
				t.Fatalf("Synchronize after %s: %v", p.name, err)
			}
		}
	}

	got := mb.Slice()
	for i := range got {
		if got[i] != float32(i) {
			t.Fatalf("managed[%d] = %v, want %v", i, got[i], float32(i))
		}
	}
}
