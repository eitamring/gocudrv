package cuda

import (
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/argpack"
)

type surfaceCalls struct {
	array3DDesc   atomic.Value
	resDesc       atomic.Value
	surfDestroys  atomic.Int32
	destroyedSurf atomic.Uint64
}

const surfArrayHandle = 0xA3D

func surfaceFixture(t *testing.T) (*Context, *surfaceCalls) {
	t.Helper()
	var c surfaceCalls
	drv := fakeMemoryDriver(&memCalls{}, 0x60000)
	drv.CuArrayCreate = func(h *cudasys.CUarray, _ *cudasys.CUDA_ARRAY_DESCRIPTOR) cudasys.CUresult {
		*h = 0xF1A7
		return cudasys.CUDA_SUCCESS
	}
	drv.CuArray3DCreate = func(h *cudasys.CUarray, desc *cudasys.CUDA_ARRAY3D_DESCRIPTOR) cudasys.CUresult {
		c.array3DDesc.Store(*desc)
		*h = surfArrayHandle
		return cudasys.CUDA_SUCCESS
	}
	drv.CuArrayDestroy = func(cudasys.CUarray) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuSurfObjectCreate = func(h *cudasys.CUsurfObject, res *cudasys.CUDA_RESOURCE_DESC) cudasys.CUresult {
		c.resDesc.Store(*res)
		*h = 0x5F5F
		return cudasys.CUDA_SUCCESS
	}
	drv.CuSurfObjectDestroy = func(h cudasys.CUsurfObject) cudasys.CUresult {
		c.surfDestroys.Add(1)
		c.destroyedSurf.Store(uint64(h))
		return cudasys.CUDA_SUCCESS
	}
	return newTestContext(t, drv), &c
}

func TestAllocArray2DWithSurfaceStore(t *testing.T) {
	ctx, c := surfaceFixture(t)
	arr, err := AllocArray2D[uint32](ctx, 64, 8, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	desc := c.array3DDesc.Load().(cudasys.CUDA_ARRAY3D_DESCRIPTOR)
	if desc.Width != 64 || desc.Height != 8 || desc.Depth != 0 {
		t.Errorf("desc dims = %d/%d/%d, want 64/8/0", desc.Width, desc.Height, desc.Depth)
	}
	if desc.Format != cudasys.AdFormatUnsignedInt32 || desc.NumChannels != 1 {
		t.Errorf("desc format = %#x x%d", desc.Format, desc.NumChannels)
	}
	if desc.Flags != cudasys.ArraySurfaceLoadStore {
		t.Errorf("flags = %#x, want surface load/store", desc.Flags)
	}
	if arr.Raw() != surfArrayHandle {
		t.Errorf("Raw = %#x, want %#x", arr.Raw(), surfArrayHandle)
	}
}

func TestAllocArray2DWithSurfaceStoreSymbolUnavailable(t *testing.T) {
	ctx, _ := surfaceFixture(t)
	ctx.driver.CuArray3DCreate = nil
	if _, err := AllocArray2D[uint32](ctx, 4, 4, WithSurfaceStore()); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing cuArray3DCreate = %v, want ErrSymbolUnavailable", err)
	}
	if _, err := AllocArray2D[uint32](ctx, 4, 4); err != nil {
		t.Errorf("plain alloc without the 3D symbol = %v, want nil", err)
	}
}

func TestNewSurface(t *testing.T) {
	ctx, c := surfaceFixture(t)
	arr, err := AllocArray2D[uint32](ctx, 16, 16, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	s, err := NewSurface(arr)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	if s.Raw() != 0x5F5F {
		t.Errorf("Raw = %#x, want 0x5F5F", s.Raw())
	}
	res := c.resDesc.Load().(cudasys.CUDA_RESOURCE_DESC)
	if res.ResType != cudasys.ResourceTypeArray || res.Handle != surfArrayHandle {
		t.Errorf("resource desc = %+v", res)
	}
}

func TestNewSurfaceRejects(t *testing.T) {
	ctx, _ := surfaceFixture(t)
	if _, err := NewSurface[uint32](nil); !errors.Is(err, ErrNilArray) {
		t.Errorf("nil array = %v, want ErrNilArray", err)
	}
	plain, err := AllocArray2D[uint32](ctx, 4, 4)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = plain.Close() })
	if _, err := NewSurface(plain); !errors.Is(err, ErrNoSurfaceStore) {
		t.Errorf("plain array = %v, want ErrNoSurfaceStore", err)
	}
	arr, err := AllocArray2D[uint32](ctx, 4, 4, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	ctx.driver.CuSurfObjectCreate = nil
	if _, err := NewSurface(arr); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing symbol = %v, want ErrSymbolUnavailable", err)
	}
	if err := arr.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := NewSurface(arr); !errors.Is(err, ErrArrayClosed) {
		t.Errorf("closed array = %v, want ErrArrayClosed", err)
	}
}

func TestSurfaceClose(t *testing.T) {
	ctx, c := surfaceFixture(t)
	arr, err := AllocArray2D[uint32](ctx, 4, 4, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	s, err := NewSurface(arr)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	ctx.driver.CuSurfObjectDestroy = func(cudasys.CUsurfObject) cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_VALUE }
	if err := s.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("failed Close = %v, want ErrInvalidValue", err)
	}
	ctx.driver.CuSurfObjectDestroy = func(h cudasys.CUsurfObject) cudasys.CUresult {
		c.destroyedSurf.Store(uint64(h))
		return cudasys.CUDA_SUCCESS
	}
	if err := s.Close(); err != nil {
		t.Errorf("retry Close = %v, want nil", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close = %v, want nil", err)
	}
	if c.destroyedSurf.Load() != 0x5F5F {
		t.Errorf("destroyed = %#x, want 0x5F5F", c.destroyedSurf.Load())
	}
}

func TestArgSurface(t *testing.T) {
	ctx, _ := surfaceFixture(t)
	arr, err := AllocArray2D[uint32](ctx, 4, 4, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	s, err := NewSurface(arr)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}

	var pk argpack.Builder
	b := kernelArgBuilder{ctx: ctx, packed: &pk}
	if err := ArgSurface(s).appendKernelArg(&b); err != nil {
		t.Fatalf("appendKernelArg: %v", err)
	}
	if err := ArgSurface(s).appendKernelArg(&b); err != nil {
		t.Fatalf("duplicate appendKernelArg: %v", err)
	}
	if b.lockCount != 1 {
		t.Errorf("lockCount = %d, want 1 (duplicate surface must not re-lock)", b.lockCount)
	}
	params := unsafe.Slice(pk.Params(), pk.Len())
	if got := *(*uint64)(params[0]); got != 0x5F5F {
		t.Errorf("packed handle = %#x, want 0x5F5F", got)
	}
	b.release()

	otherCtx, _ := surfaceFixture(t)
	b2 := kernelArgBuilder{ctx: otherCtx, packed: &argpack.Builder{}}
	if err := ArgSurface(s).appendKernelArg(&b2); !errors.Is(err, ErrContextMismatch) {
		t.Errorf("wrong context = %v, want ErrContextMismatch", err)
	}

	var nilSurf *Surface
	b3 := kernelArgBuilder{packed: &argpack.Builder{}}
	if err := ArgSurface(nilSurf).appendKernelArg(&b3); !errors.Is(err, ErrNilSurface) {
		t.Errorf("nil surface = %v, want ErrNilSurface", err)
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	b4 := kernelArgBuilder{packed: &argpack.Builder{}}
	if err := ArgSurface(s).appendKernelArg(&b4); !errors.Is(err, ErrSurfaceClosed) {
		t.Errorf("closed surface = %v, want ErrSurfaceClosed", err)
	}
}

func TestPackArgSurface(t *testing.T) {
	ctx, _ := surfaceFixture(t)
	arr, err := AllocArray2D[uint32](ctx, 4, 4, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	s, err := NewSurface(arr)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	p, err := Pack(ArgSurface(s), ArgValue(int32(4)))
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if p.Len() != 2 {
		t.Errorf("Len = %d, want 2", p.Len())
	}
	params := unsafe.Slice(p.packed.Params(), p.Len())
	if got := *(*uint64)(params[0]); got != 0x5F5F {
		t.Errorf("packed handle = %#x, want 0x5F5F", got)
	}
	if err := s.Close(); err != nil {
		t.Errorf("Close after Pack (snapshot takes no lock) = %v", err)
	}
}

func TestAllocArray2DNilOption(t *testing.T) {
	ctx, _ := surfaceFixture(t)
	arr, err := AllocArray2D[uint32](ctx, 4, 4, nil, WithSurfaceStore(), nil)
	if err != nil {
		t.Fatalf("AllocArray2D with nil options: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	if !arr.surface {
		t.Error("surface store not applied alongside nil options")
	}
}

func TestNewTextureOnSurfaceArray(t *testing.T) {
	ctx, _ := surfaceFixture(t)
	ctx.driver.CuTexObjectCreate = func(h *cudasys.CUtexObject, _ *cudasys.CUDA_RESOURCE_DESC, _ *cudasys.CUDA_TEXTURE_DESC, _ unsafe.Pointer) cudasys.CUresult {
		*h = 0x7E7E
		return cudasys.CUDA_SUCCESS
	}
	ctx.driver.CuTexObjectDestroy = func(cudasys.CUtexObject) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	arr, err := AllocArray2D[uint32](ctx, 4, 4, WithSurfaceStore())
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture over a surface-store array: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Errorf("tex Close: %v", err)
	}
}

func TestSurfaceNilReceivers(t *testing.T) {
	var s *Surface
	if s.Raw() != 0 {
		t.Error("nil surface Raw must be zero")
	}
	if err := s.Close(); !errors.Is(err, ErrNilSurface) {
		t.Errorf("nil surface Close = %v, want ErrNilSurface", err)
	}
}
