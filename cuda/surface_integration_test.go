//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

func TestRealSurfaceWrite(t *testing.T) {
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

	const w, h = 8, 4
	const n = w * h

	arr, err := AllocArray2D[uint32](ctx, w, h, WithSurfaceStore())
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks 3D array symbols")
	}
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })

	surf, err := NewSurface(arr)
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks surface object symbols")
	}
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	t.Cleanup(func() { _ = surf.Close() })

	mod, err := ctx.LoadModuleFromFile("testdata/surf_write.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("surf_write")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	bg := context.Background()
	if err := fn.Launch(bg, LaunchConfig1D(n, 32),
		ArgSurface(surf),
		ArgValue(int32(w)),
		ArgValue(int32(n)),
	); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]uint32, n)
	if err := arr.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range got {
		if got[i] != uint32(i)*3 {
			t.Fatalf("texel %d = %d, want %d", i, got[i], i*3)
		}
	}

	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture over the surface-store array: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Fatalf("tex Close: %v", err)
	}
	t.Logf("kernel wrote all %d texels through the surface object correctly", n)
}
