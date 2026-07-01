//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

func TestRealVolumeRoundTrip(t *testing.T) {
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

	const w, h, d = 17, 5, 3
	n := w * h * d
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 1.5
	}

	v, err := AllocVolume[float32](ctx, w, h, d)
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks 3D copy symbols")
	}
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = v.Close() })
	if v.Pitch() < uint64(w)*4 {
		t.Fatalf("pitch %d smaller than a row (%d bytes)", v.Pitch(), w*4)
	}

	bg := context.Background()
	if err := v.CopyFrom(bg, src); errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks cuMemcpy3D")
	} else if err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	got := make([]float32, n)
	if err := v.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("element %d = %v, want %v", i, got[i], src[i])
		}
	}
	t.Logf("round-tripped a %dx%dx%d volume (pitch %d bytes)", w, h, d, v.Pitch())
}
