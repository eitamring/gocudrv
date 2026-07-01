package cuda

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

type volumeCalls struct {
	allocW, allocH                                  atomic.Uint64
	allocESB                                        atomic.Uint32
	copies                                          atomic.Int32
	srcType, dstType                                atomic.Uint32
	srcPitch, dstPitch, srcHeight, dstHeight        atomic.Uint64
	widthBytes, height, depth, srcDevice, dstDevice atomic.Uint64
}

const volumeBase = 0x40000

func volumeFixture(t *testing.T) (*Context, *volumeCalls) {
	t.Helper()
	var c volumeCalls
	drv := fakeMemoryDriver(&memCalls{}, volumeBase)
	drv.CuMemAllocPitch = func(ptr *cudasys.CUdeviceptr, pitch *uint64, w, h uint64, esb uint32) cudasys.CUresult {
		c.allocW.Store(w)
		c.allocH.Store(h)
		c.allocESB.Store(esb)
		*ptr = cudasys.CUdeviceptr(volumeBase)
		*pitch = 512
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpy3D = func(d *cudasys.Memcpy3D) cudasys.CUresult {
		c.copies.Add(1)
		c.srcType.Store(d.SrcMemoryType)
		c.dstType.Store(d.DstMemoryType)
		c.srcPitch.Store(d.SrcPitch)
		c.dstPitch.Store(d.DstPitch)
		c.srcHeight.Store(d.SrcHeight)
		c.dstHeight.Store(d.DstHeight)
		c.widthBytes.Store(d.WidthInBytes)
		c.height.Store(d.Height)
		c.depth.Store(d.Depth)
		c.srcDevice.Store(uint64(d.SrcDevice))
		c.dstDevice.Store(uint64(d.DstDevice))
		return cudasys.CUDA_SUCCESS
	}
	return newTestContext(t, drv), &c
}

func TestAllocVolume(t *testing.T) {
	ctx, c := volumeFixture(t)
	v, err := AllocVolume[float32](ctx, 64, 8, 4)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = v.Close() })
	if c.allocW.Load() != 64*4 || c.allocH.Load() != 8*4 || c.allocESB.Load() != 4 {
		t.Errorf("alloc args = (%d,%d,%d), want (256,32,4)", c.allocW.Load(), c.allocH.Load(), c.allocESB.Load())
	}
	if v.Width() != 64 || v.Height() != 8 || v.Depth() != 4 || v.Pitch() != 512 {
		t.Errorf("W/H/D/Pitch = %d/%d/%d/%d, want 64/8/4/512", v.Width(), v.Height(), v.Depth(), v.Pitch())
	}
	if v.DevicePtr() != volumeBase {
		t.Errorf("DevicePtr = %#x, want %#x", v.DevicePtr(), volumeBase)
	}

	d8, err := AllocVolume[float64](ctx, 4, 2, 2)
	if err != nil {
		t.Fatalf("AllocVolume[float64]: %v", err)
	}
	t.Cleanup(func() { _ = d8.Close() })
	if c.allocESB.Load() != 8 {
		t.Errorf("float64 elementSizeBytes = %d, want 8", c.allocESB.Load())
	}
}

func TestAllocVolumeRejects(t *testing.T) {
	ctx, _ := volumeFixture(t)
	if _, err := AllocVolume[float32](nil, 4, 4, 4); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	if _, err := AllocVolume[float32](ctx, 0, 4, 4); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero width = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocVolume[float32](ctx, 4, 4, -1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("negative depth = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocVolume[float32](ctx, math.MaxInt, 2, 2); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("element-count overflow = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocVolume[float64](ctx, math.MaxInt, 1, 1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("row-byte overflow = %v, want ErrInvalidLength", err)
	}
}

func TestVolumeCopyFromAndTo(t *testing.T) {
	ctx, c := volumeFixture(t)
	v, err := AllocVolume[float32](ctx, 64, 8, 4)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = v.Close() })
	bg := context.Background()

	if err := v.CopyFrom(bg, make([]float32, 64*8*4)); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	if c.copies.Load() != 1 {
		t.Errorf("copies = %d, want 1", c.copies.Load())
	}
	if c.srcType.Load() != cudasys.MemoryTypeHost || c.dstType.Load() != cudasys.MemoryTypeDevice {
		t.Errorf("HtoD mem types = %d->%d", c.srcType.Load(), c.dstType.Load())
	}
	if c.dstPitch.Load() != 512 || c.widthBytes.Load() != 256 || c.height.Load() != 8 || c.depth.Load() != 4 {
		t.Errorf("desc = dstPitch %d width %d height %d depth %d, want 512/256/8/4", c.dstPitch.Load(), c.widthBytes.Load(), c.height.Load(), c.depth.Load())
	}
	if c.srcPitch.Load() != 256 || c.srcHeight.Load() != 8 || c.dstHeight.Load() != 8 {
		t.Errorf("desc = srcPitch %d srcHeight %d dstHeight %d, want 256/8/8", c.srcPitch.Load(), c.srcHeight.Load(), c.dstHeight.Load())
	}
	if c.dstDevice.Load() != volumeBase {
		t.Errorf("dst device = %#x, want %#x", c.dstDevice.Load(), volumeBase)
	}

	if err := v.CopyTo(bg, make([]float32, 64*8*4)); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	if c.srcType.Load() != cudasys.MemoryTypeDevice || c.dstType.Load() != cudasys.MemoryTypeHost {
		t.Errorf("DtoH mem types = %d->%d", c.srcType.Load(), c.dstType.Load())
	}
	if c.srcPitch.Load() != 512 || c.dstPitch.Load() != 256 {
		t.Errorf("DtoH pitches = src %d dst %d, want 512/256", c.srcPitch.Load(), c.dstPitch.Load())
	}
	if c.srcHeight.Load() != 8 || c.dstHeight.Load() != 8 || c.srcDevice.Load() != volumeBase {
		t.Errorf("DtoH desc = srcHeight %d dstHeight %d srcDevice %#x, want 8/8/%#x", c.srcHeight.Load(), c.dstHeight.Load(), c.srcDevice.Load(), volumeBase)
	}

	if err := v.CopyFrom(bg, make([]float32, 10)); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("wrong-length CopyFrom = %v, want ErrLengthMismatch", err)
	}
}

func TestVolumeSymbolUnavailable(t *testing.T) {
	ctx, _ := volumeFixture(t)
	ctx.driver.CuMemAllocPitch = nil
	if _, err := AllocVolume[float32](ctx, 4, 4, 4); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("AllocVolume = %v, want ErrSymbolUnavailable", err)
	}
}

func TestVolumeCopySymbolUnavailable(t *testing.T) {
	ctx, _ := volumeFixture(t)
	v, err := AllocVolume[float32](ctx, 8, 8, 2)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = v.Close() })
	ctx.driver.CuMemcpy3D = nil
	if err := v.CopyFrom(context.Background(), make([]float32, 8*8*2)); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("CopyFrom = %v, want ErrSymbolUnavailable", err)
	}
}

func TestVolumeCloseRetry(t *testing.T) {
	ctx, _ := volumeFixture(t)
	v, err := AllocVolume[float32](ctx, 8, 8, 2)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	ctx.driver.CuMemFree = func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_VALUE }
	if err := v.Close(); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("failed Close = %v, want ErrInvalidValue", err)
	}
	ctx.driver.CuMemFree = func(cudasys.CUdeviceptr) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	if err := v.Close(); err != nil {
		t.Errorf("retry Close = %v, want nil", err)
	}
}

func TestVolumeNilReceiver(t *testing.T) {
	var v *Volume[float32]
	if v.Width() != 0 || v.Height() != 0 || v.Depth() != 0 || v.Pitch() != 0 || v.DevicePtr() != 0 {
		t.Error("nil receiver accessors must be zero")
	}
	if err := v.CopyFrom(context.Background(), nil); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil CopyFrom = %v, want ErrNilBuffer", err)
	}
	if err := v.Close(); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil Close = %v, want ErrNilBuffer", err)
	}
}

func TestVolumeCloseIdempotent(t *testing.T) {
	ctx, _ := volumeFixture(t)
	v, err := AllocVolume[float32](ctx, 8, 8, 2)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	if err := v.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := v.Close(); err != nil {
		t.Errorf("second Close = %v, want nil", err)
	}
	if err := v.CopyFrom(context.Background(), make([]float32, 8*8*2)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyFrom after close = %v, want ErrBufferClosed", err)
	}
}
