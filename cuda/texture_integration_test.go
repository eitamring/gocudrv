//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

func TestRealArrayAndTexture(t *testing.T) {
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

	const w, h = 33, 9
	src := make([]float32, w*h)
	for i := range src {
		src[i] = float32(i) * 0.25
	}

	arr, err := AllocArray2D[float32](ctx, w, h)
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks CUDA array symbols")
	}
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })

	bg := context.Background()
	if err := arr.CopyFrom(bg, src); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	got := make([]float32, w*h)
	if err := arr.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("element %d = %v, want %v", i, got[i], src[i])
		}
	}

	tex, err := NewTexture(arr, TextureConfig{AddressMode: AddressClamp, FilterMode: FilterPoint})
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks texture object symbols")
	}
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	if tex.Raw() == 0 {
		t.Fatal("texture handle is zero")
	}
	if err := tex.Close(); err != nil {
		t.Fatalf("tex Close: %v", err)
	}
	t.Logf("round-tripped a %dx%d array and created a texture object over it", w, h)
}

func TestRealTextureSample(t *testing.T) {
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
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*3 + 0.5
	}

	arr, err := AllocArray2D[float32](ctx, w, h)
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks CUDA array symbols")
	}
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	bg := context.Background()
	if err := arr.CopyFrom(bg, src); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	tex, err := NewTexture(arr, TextureConfig{AddressMode: AddressClamp, FilterMode: FilterPoint})
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks texture object symbols")
	}
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	t.Cleanup(func() { _ = tex.Close() })

	out, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	t.Cleanup(func() { _ = out.Close() })

	mod, err := ctx.LoadModuleFromFile("testdata/tex_sample.ptx")
	if err != nil {
		t.Fatalf("LoadModuleFromFile: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("tex_sample")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	if err := fn.Launch(bg, LaunchConfig1D(n, 32),
		ArgTexture(tex),
		Arg(out),
		ArgValue(int32(w)),
		ArgValue(int32(n)),
	); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, n)
	if err := out.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("sampled[%d] = %v, want %v", i, got[i], src[i])
		}
	}
	t.Logf("kernel sampled all %d texels through the texture object correctly", n)
}
