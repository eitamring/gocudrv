package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/argpack"
)

type textureCalls struct {
	arrayCreates, arrayDestroys, texCreates, texDestroys atomic.Int32
	copies                                               atomic.Int32
	arrayDesc                                            atomic.Value
	resDesc, texDesc                                     atomic.Value
	copyDesc                                             atomic.Value
	destroyedArray                                       atomic.Uintptr
	destroyedTex                                         atomic.Uint64
}

const arrayHandle = 0xA11A7

func textureFixture(t *testing.T) (*Context, *textureCalls) {
	t.Helper()
	var c textureCalls
	drv := fakeMemoryDriver(&memCalls{}, 0x50000)
	drv.CuArrayCreate = func(h *cudasys.CUarray, desc *cudasys.CUDA_ARRAY_DESCRIPTOR) cudasys.CUresult {
		c.arrayCreates.Add(1)
		c.arrayDesc.Store(*desc)
		*h = arrayHandle
		return cudasys.CUDA_SUCCESS
	}
	drv.CuArrayDestroy = func(h cudasys.CUarray) cudasys.CUresult {
		c.arrayDestroys.Add(1)
		c.destroyedArray.Store(uintptr(h))
		return cudasys.CUDA_SUCCESS
	}
	drv.CuTexObjectCreate = func(h *cudasys.CUtexObject, res *cudasys.CUDA_RESOURCE_DESC, tex *cudasys.CUDA_TEXTURE_DESC, view unsafe.Pointer) cudasys.CUresult {
		c.texCreates.Add(1)
		c.resDesc.Store(*res)
		c.texDesc.Store(*tex)
		if view != nil {
			t.Error("resource-view desc must be nil")
		}
		*h = 0x7E7E
		return cudasys.CUDA_SUCCESS
	}
	drv.CuTexObjectDestroy = func(h cudasys.CUtexObject) cudasys.CUresult {
		c.texDestroys.Add(1)
		c.destroyedTex.Store(uint64(h))
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpy2D = func(d *cudasys.Memcpy2D) cudasys.CUresult {
		c.copies.Add(1)
		c.copyDesc.Store(*d)
		return cudasys.CUDA_SUCCESS
	}
	return newTestContext(t, drv), &c
}

func TestAllocArray2D(t *testing.T) {
	ctx, c := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	desc := c.arrayDesc.Load().(cudasys.CUDA_ARRAY_DESCRIPTOR)
	if desc.Width != 64 || desc.Height != 8 || desc.Format != cudasys.AdFormatFloat || desc.NumChannels != 1 {
		t.Errorf("desc = %+v, want 64x8 float x1", desc)
	}
	if arr.Width() != 64 || arr.Height() != 8 || arr.Raw() != arrayHandle {
		t.Errorf("accessors = %d/%d/%#x", arr.Width(), arr.Height(), arr.Raw())
	}

	if _, err := AllocArray2D[uint8](ctx, 4, 4); err != nil {
		t.Fatalf("AllocArray2D[uint8]: %v", err)
	}
	desc = c.arrayDesc.Load().(cudasys.CUDA_ARRAY_DESCRIPTOR)
	if desc.Format != cudasys.AdFormatUnsignedInt8 {
		t.Errorf("uint8 format = %#x, want %#x", desc.Format, cudasys.AdFormatUnsignedInt8)
	}
}

func TestAllocArray2DRejects(t *testing.T) {
	ctx, c := textureFixture(t)
	if _, err := AllocArray2D[float32](nil, 4, 4); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	if _, err := AllocArray2D[float32](ctx, 0, 4); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero width = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocArray2D[float32](ctx, 4, -1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("negative height = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocArray2D[float64](ctx, 4, 4); !errors.Is(err, ErrUnsupportedElement) {
		t.Errorf("float64 = %v, want ErrUnsupportedElement", err)
	}
	if _, err := AllocArray2D[int64](ctx, 4, 4); !errors.Is(err, ErrUnsupportedElement) {
		t.Errorf("int64 = %v, want ErrUnsupportedElement", err)
	}
	if c.arrayCreates.Load() != 0 {
		t.Errorf("array creates on rejected calls = %d, want 0", c.arrayCreates.Load())
	}
	ctx.driver.CuArrayCreate = nil
	if _, err := AllocArray2D[float32](ctx, 4, 4); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing symbol = %v, want ErrSymbolUnavailable", err)
	}
}

func TestArray2DCopyFromAndTo(t *testing.T) {
	ctx, c := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	bg := context.Background()

	if err := arr.CopyFrom(bg, make([]float32, 64*8)); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	d := c.copyDesc.Load().(cudasys.Memcpy2D)
	if d.SrcMemoryType != cudasys.MemoryTypeHost || d.DstMemoryType != cudasys.MemoryTypeArray {
		t.Errorf("HtoA mem types = %d->%d", d.SrcMemoryType, d.DstMemoryType)
	}
	if d.DstArray != arrayHandle || d.SrcPitch != 256 || d.WidthInBytes != 256 || d.Height != 8 {
		t.Errorf("HtoA desc = %+v", d)
	}

	if err := arr.CopyTo(bg, make([]float32, 64*8)); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	d = c.copyDesc.Load().(cudasys.Memcpy2D)
	if d.SrcMemoryType != cudasys.MemoryTypeArray || d.DstMemoryType != cudasys.MemoryTypeHost {
		t.Errorf("AtoH mem types = %d->%d", d.SrcMemoryType, d.DstMemoryType)
	}
	if d.SrcArray != arrayHandle || d.DstPitch != 256 {
		t.Errorf("AtoH desc = %+v", d)
	}

	if err := arr.CopyFrom(bg, make([]float32, 3)); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("wrong-length CopyFrom = %v, want ErrLengthMismatch", err)
	}
}

func TestNewTexture(t *testing.T) {
	ctx, c := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })

	tex, err := NewTexture(arr, TextureConfig{
		AddressMode:           AddressClamp,
		FilterMode:            FilterLinear,
		NormalizedCoordinates: true,
	})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	if tex.Raw() != 0x7E7E {
		t.Errorf("Raw = %#x, want 0x7E7E", tex.Raw())
	}
	res := c.resDesc.Load().(cudasys.CUDA_RESOURCE_DESC)
	if res.ResType != cudasys.ResourceTypeArray || res.Handle != arrayHandle {
		t.Errorf("resource desc = %+v", res)
	}
	td := c.texDesc.Load().(cudasys.CUDA_TEXTURE_DESC)
	if td.AddressMode != [3]uint32{1, 1, 1} || td.FilterMode != cudasys.FilterModeLinear {
		t.Errorf("texture desc = %+v", td)
	}
	if td.Flags != cudasys.TexNormalizedCoordinate {
		t.Errorf("flags = %#x, want normalized only", td.Flags)
	}
}

func TestNewTextureIntegerReadFlag(t *testing.T) {
	ctx, c := textureFixture(t)
	arr, err := AllocArray2D[uint8](ctx, 16, 16)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	td := c.texDesc.Load().(cudasys.CUDA_TEXTURE_DESC)
	if td.Flags != cudasys.TexReadAsInteger {
		t.Errorf("flags = %#x, want read-as-integer", td.Flags)
	}
}

func TestNewTextureRejects(t *testing.T) {
	ctx, _ := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 4, 4)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	if _, err := NewTexture[float32](nil, TextureConfig{}); !errors.Is(err, ErrNilArray) {
		t.Errorf("nil array = %v, want ErrNilArray", err)
	}
	ctx.driver.CuTexObjectCreate = nil
	if _, err := NewTexture(arr, TextureConfig{}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing symbol = %v, want ErrSymbolUnavailable", err)
	}
	if err := arr.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := NewTexture(arr, TextureConfig{}); !errors.Is(err, ErrArrayClosed) {
		t.Errorf("closed array = %v, want ErrArrayClosed", err)
	}
}

func TestArrayAndTextureClose(t *testing.T) {
	ctx, c := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 4, 4)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}

	if err := tex.Close(); err != nil {
		t.Fatalf("tex Close: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Errorf("second tex Close = %v, want nil", err)
	}
	if c.destroyedTex.Load() != 0x7E7E || c.texDestroys.Load() != 1 {
		t.Errorf("tex destroys = %d handle %#x", c.texDestroys.Load(), c.destroyedTex.Load())
	}

	ctx.driver.CuArrayDestroy = func(cudasys.CUarray) cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_VALUE }
	if err := arr.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("failed Close = %v, want ErrInvalidValue", err)
	}
	ctx.driver.CuArrayDestroy = func(h cudasys.CUarray) cudasys.CUresult {
		c.destroyedArray.Store(uintptr(h))
		return cudasys.CUDA_SUCCESS
	}
	if err := arr.Close(); err != nil {
		t.Errorf("retry Close = %v, want nil", err)
	}
	if c.destroyedArray.Load() != arrayHandle {
		t.Errorf("destroyed array = %#x, want %#x", c.destroyedArray.Load(), arrayHandle)
	}
	if err := arr.CopyFrom(context.Background(), make([]float32, 16)); !errors.Is(err, ErrArrayClosed) {
		t.Errorf("CopyFrom after close = %v, want ErrArrayClosed", err)
	}
}

func TestArgTexture(t *testing.T) {
	ctx, _ := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 4, 4)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}

	var pk argpack.Builder
	b := kernelArgBuilder{ctx: ctx, packed: &pk}
	if err := ArgTexture(tex).appendKernelArg(&b); err != nil {
		t.Fatalf("appendKernelArg: %v", err)
	}
	params := unsafe.Slice(pk.Params(), pk.Len())
	if got := *(*uint64)(params[0]); got != 0x7E7E {
		t.Errorf("packed handle = %#x, want 0x7E7E", got)
	}
	b.release()

	otherCtx, _ := textureFixture(t)
	b2 := kernelArgBuilder{ctx: otherCtx, packed: &argpack.Builder{}}
	if err := ArgTexture(tex).appendKernelArg(&b2); !errors.Is(err, ErrContextMismatch) {
		t.Errorf("wrong context = %v, want ErrContextMismatch", err)
	}

	var nilTex *Texture
	b3 := kernelArgBuilder{packed: &argpack.Builder{}}
	if err := ArgTexture(nilTex).appendKernelArg(&b3); !errors.Is(err, ErrNilTexture) {
		t.Errorf("nil texture = %v, want ErrNilTexture", err)
	}

	if err := tex.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	b4 := kernelArgBuilder{packed: &argpack.Builder{}}
	if err := ArgTexture(tex).appendKernelArg(&b4); !errors.Is(err, ErrTextureClosed) {
		t.Errorf("closed texture = %v, want ErrTextureClosed", err)
	}
}

func TestArgTextureDuplicateSharesLock(t *testing.T) {
	ctx, _ := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 4, 4)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	t.Cleanup(func() { _ = tex.Close() })

	var pk argpack.Builder
	b := kernelArgBuilder{ctx: ctx, packed: &pk}
	if err := ArgTexture(tex).appendKernelArg(&b); err != nil {
		t.Fatalf("first appendKernelArg: %v", err)
	}
	if err := ArgTexture(tex).appendKernelArg(&b); err != nil {
		t.Fatalf("second appendKernelArg: %v", err)
	}
	if b.lockCount != 1 {
		t.Errorf("lockCount = %d, want 1 (duplicate texture must not re-lock)", b.lockCount)
	}
	if pk.Len() != 2 {
		t.Errorf("packed args = %d, want 2", pk.Len())
	}
	b.release()
	if err := tex.Close(); err != nil {
		t.Errorf("Close after release = %v", err)
	}
}

func TestPackArgTexture(t *testing.T) {
	ctx, _ := textureFixture(t)
	arr, err := AllocArray2D[float32](ctx, 4, 4)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = arr.Close() })
	tex, err := NewTexture(arr, TextureConfig{})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	t.Cleanup(func() { _ = tex.Close() })

	p, err := Pack(ArgTexture(tex), ArgValue(int32(4)))
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if p.Len() != 2 {
		t.Errorf("Len = %d, want 2", p.Len())
	}
	params := unsafe.Slice(p.packed.Params(), p.Len())
	if got := *(*uint64)(params[0]); got != 0x7E7E {
		t.Errorf("packed handle = %#x, want 0x7E7E", got)
	}
	if err := tex.Close(); err != nil {
		t.Errorf("Close after Pack (snapshot takes no lock) = %v", err)
	}
}

func TestTextureNilReceivers(t *testing.T) {
	var arr *Array2D[float32]
	if arr.Width() != 0 || arr.Height() != 0 || arr.Raw() != 0 {
		t.Error("nil array accessors must be zero")
	}
	if err := arr.CopyFrom(context.Background(), nil); !errors.Is(err, ErrNilArray) {
		t.Errorf("nil CopyFrom = %v, want ErrNilArray", err)
	}
	if err := arr.Close(); !errors.Is(err, ErrNilArray) {
		t.Errorf("nil array Close = %v, want ErrNilArray", err)
	}
	var tex *Texture
	if tex.Raw() != 0 {
		t.Error("nil texture Raw must be zero")
	}
	if err := tex.Close(); !errors.Is(err, ErrNilTexture) {
		t.Errorf("nil texture Close = %v, want ErrNilTexture", err)
	}
}
