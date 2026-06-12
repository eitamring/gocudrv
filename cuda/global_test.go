package cuda

import (
	"context"
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

const testGlobalPtr = 0xD0D0
const testGlobalBytes = 16 // room for 4 float32

func globalTestModule(t *testing.T) (*Context, *Module) {
	t.Helper()
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)
	mod, err := ctx.LoadModule([]byte{'P', 'T', 'X', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	ctx.driver.CuModuleGetGlobal = func(dptr *cudasys.CUdeviceptr, bytes *uint64, _ cudasys.CUmodule, _ *byte) cudasys.CUresult {
		*dptr = testGlobalPtr
		*bytes = testGlobalBytes
		return cudasys.CUDA_SUCCESS
	}
	return ctx, mod
}

func TestModuleGlobalHappy(t *testing.T) {
	_, mod := globalTestModule(t)
	g, err := mod.Global("counter")
	if err != nil {
		t.Fatalf("Global: %v", err)
	}
	if g.Bytes() != testGlobalBytes {
		t.Errorf("Bytes = %d, want %d", g.Bytes(), testGlobalBytes)
	}
	if g.Name() != "counter" {
		t.Errorf("Name = %q, want counter", g.Name())
	}
}

func TestModuleGlobalRejects(t *testing.T) {
	ctx, mod := globalTestModule(t)

	closedMod, err := ctx.LoadModule([]byte{'P', 0})
	if err != nil {
		t.Fatalf("LoadModule closedMod: %v", err)
	}
	if err := closedMod.Close(); err != nil {
		t.Fatalf("close closedMod: %v", err)
	}

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil module", func() error { var m *Module; _, e := m.Global("x"); return e }, ErrNilModule},
		{"empty name", func() error { _, e := mod.Global(""); return e }, ErrEmptyGlobalName},
		{"embedded null", func() error { _, e := mod.Global("a\x00b"); return e }, ErrInvalidGlobalName},
		{"closed module", func() error { _, e := closedMod.Global("x"); return e }, ErrModuleClosed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestModuleGlobalPropagatesError(t *testing.T) {
	ctx, mod := globalTestModule(t)
	ctx.driver.CuModuleGetGlobal = func(*cudasys.CUdeviceptr, *uint64, cudasys.CUmodule, *byte) cudasys.CUresult {
		return cudasys.CUDA_ERROR_NOT_FOUND
	}
	if _, err := mod.Global("missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestWriteGlobal(t *testing.T) {
	ctx, mod := globalTestModule(t)
	g, err := mod.Global("counter")
	if err != nil {
		t.Fatalf("Global: %v", err)
	}
	var gotPtr cudasys.CUdeviceptr
	var gotBytes uint64
	calls := 0
	ctx.driver.CuMemcpyHtoD = func(dst cudasys.CUdeviceptr, _ *byte, bytes uint64) cudasys.CUresult {
		calls++
		gotPtr, gotBytes = dst, bytes
		return cudasys.CUDA_SUCCESS
	}
	if err := WriteGlobal(context.Background(), g, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("WriteGlobal: %v", err)
	}
	if calls != 1 || gotPtr != testGlobalPtr || gotBytes != 16 {
		t.Errorf("got calls=%d ptr=%#x bytes=%d, want 1, %#x, 16", calls, gotPtr, gotBytes, testGlobalPtr)
	}
}

func TestReadGlobal(t *testing.T) {
	ctx, mod := globalTestModule(t)
	g, err := mod.Global("counter")
	if err != nil {
		t.Fatalf("Global: %v", err)
	}
	var gotBytes uint64
	calls := 0
	ctx.driver.CuMemcpyDtoH = func(_ *byte, src cudasys.CUdeviceptr, bytes uint64) cudasys.CUresult {
		calls++
		gotBytes = bytes
		if src != testGlobalPtr {
			t.Errorf("src = %#x, want %#x", src, testGlobalPtr)
		}
		return cudasys.CUDA_SUCCESS
	}
	dst := make([]float32, 4)
	if err := ReadGlobal(context.Background(), dst, g); err != nil {
		t.Fatalf("ReadGlobal: %v", err)
	}
	if calls != 1 || gotBytes != 16 {
		t.Errorf("got calls=%d bytes=%d, want 1, 16", calls, gotBytes)
	}
}

func TestGlobalReadWriteRejects(t *testing.T) {
	ctx, mod := globalTestModule(t)
	g, err := mod.Global("counter")
	if err != nil {
		t.Fatalf("Global: %v", err)
	}
	ctx.driver.CuMemcpyHtoD = func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
		t.Error("HtoD must not run on rejected input")
		return cudasys.CUDA_SUCCESS
	}
	ctx.driver.CuMemcpyDtoH = func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		t.Error("DtoH must not run on rejected input")
		return cudasys.CUDA_SUCCESS
	}
	bg := context.Background()
	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"write nil global", func() error { return WriteGlobal[float32](bg, nil, []float32{1}) }, ErrNilGlobal},
		{"write empty slice", func() error { return WriteGlobal(bg, g, []float32{}) }, ErrLengthMismatch},
		{"write too large", func() error { return WriteGlobal(bg, g, []float32{1, 2, 3, 4, 5}) }, ErrLengthMismatch},
		{"read nil global", func() error { return ReadGlobal[float32](bg, []float32{1}, nil) }, ErrNilGlobal},
		{"read empty slice", func() error { return ReadGlobal(bg, []float32{}, g) }, ErrLengthMismatch},
		{"read too large", func() error { return ReadGlobal(bg, make([]float32, 5), g) }, ErrLengthMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestGlobalAfterModuleClose(t *testing.T) {
	_, mod := globalTestModule(t)
	g, err := mod.Global("counter")
	if err != nil {
		t.Fatalf("Global: %v", err)
	}
	if err := mod.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := WriteGlobal(context.Background(), g, []float32{1, 2, 3, 4}); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("WriteGlobal after close = %v, want ErrModuleClosed", err)
	}
	if err := ReadGlobal(context.Background(), make([]float32, 4), g); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("ReadGlobal after close = %v, want ErrModuleClosed", err)
	}
}
