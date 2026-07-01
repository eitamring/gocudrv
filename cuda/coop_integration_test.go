//go:build cuda_integration

package cuda

import (
	"context"
	"testing"
)

func TestRealCooperativeLaunch(t *testing.T) {
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	if ok, err := dev.Attribute(DeviceAttributeCooperativeLaunch); err != nil {
		t.Fatalf("cooperative-launch attribute: %v", err)
	} else if ok == 0 {
		t.Skip("device does not support cooperative launch")
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

	const blockSize = 256
	maxBlocks, err := fn.MaxCooperativeGridBlocks(blockSize, 0)
	if err != nil {
		t.Fatalf("MaxCooperativeGridBlocks: %v", err)
	}
	cfg := LaunchConfig1D(n, blockSize)
	if int(cfg.GridX) > maxBlocks {
		t.Skipf("grid of %d blocks exceeds co-resident max %d", cfg.GridX, maxBlocks)
	}

	if err := fn.LaunchCooperative(bg, cfg, Arg(a), Arg(b), Arg(out), ArgValue(int32(n))); err != nil {
		t.Fatalf("LaunchCooperative: %v", err)
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
	t.Logf("cooperative launch of vector_add for %d elements, co-resident max %d blocks", n, maxBlocks)
}
