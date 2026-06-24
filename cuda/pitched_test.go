package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

type pitchCalls struct {
	allocW, allocH                         atomic.Uint64
	allocESB                               atomic.Uint32
	copies                                 atomic.Int32
	srcType, dstType                       atomic.Uint32
	srcPitch, dstPitch, widthBytes, height atomic.Uint64
	srcDevice, dstDevice                   atomic.Uint64
}

const pitchBase = 0x30000

func pitchedFixture(t *testing.T) (*Context, *pitchCalls) {
	t.Helper()
	var c pitchCalls
	drv := fakeMemoryDriver(&memCalls{}, pitchBase)
	drv.CuMemAllocPitch = func(ptr *cudasys.CUdeviceptr, pitch *uint64, w, h uint64, esb uint32) cudasys.CUresult {
		c.allocW.Store(w)
		c.allocH.Store(h)
		c.allocESB.Store(esb)
		*ptr = cudasys.CUdeviceptr(pitchBase)
		*pitch = 512
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpy2D = func(d *cudasys.Memcpy2D) cudasys.CUresult {
		c.copies.Add(1)
		c.srcType.Store(d.SrcMemoryType)
		c.dstType.Store(d.DstMemoryType)
		c.srcPitch.Store(d.SrcPitch)
		c.dstPitch.Store(d.DstPitch)
		c.widthBytes.Store(d.WidthInBytes)
		c.height.Store(d.Height)
		c.srcDevice.Store(uint64(d.SrcDevice))
		c.dstDevice.Store(uint64(d.DstDevice))
		return cudasys.CUDA_SUCCESS
	}
	return newTestContext(t, drv), &c
}

func TestAllocPitched(t *testing.T) {
	ctx, c := pitchedFixture(t)
	buf, err := AllocPitched[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("AllocPitched: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	if c.allocW.Load() != 64*4 || c.allocH.Load() != 8 || c.allocESB.Load() != 4 {
		t.Errorf("alloc args = (%d,%d,%d), want (256,8,4)", c.allocW.Load(), c.allocH.Load(), c.allocESB.Load())
	}
	if buf.Width() != 64 || buf.Height() != 8 || buf.Pitch() != 512 {
		t.Errorf("W/H/Pitch = %d/%d/%d, want 64/8/512", buf.Width(), buf.Height(), buf.Pitch())
	}
	if buf.DevicePtr() != pitchBase {
		t.Errorf("DevicePtr = %#x, want %#x", buf.DevicePtr(), pitchBase)
	}

	// float64 elements hint an 8-byte element size.
	d8, err := AllocPitched[float64](ctx, 4, 2)
	if err != nil {
		t.Fatalf("AllocPitched[float64]: %v", err)
	}
	t.Cleanup(func() { _ = d8.Close() })
	if c.allocESB.Load() != 8 {
		t.Errorf("float64 elementSizeBytes = %d, want 8", c.allocESB.Load())
	}
}

func TestAllocPitchedRejects(t *testing.T) {
	ctx, _ := pitchedFixture(t)
	if _, err := AllocPitched[float32](nil, 4, 4); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	if _, err := AllocPitched[float32](ctx, 0, 4); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero width = %v, want ErrInvalidLength", err)
	}
	if _, err := AllocPitched[float32](ctx, 4, -1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("negative height = %v, want ErrInvalidLength", err)
	}
}

func TestPitchedCopyFromAndTo(t *testing.T) {
	ctx, c := pitchedFixture(t)
	buf, err := AllocPitched[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("AllocPitched: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	bg := context.Background()

	if err := buf.CopyFrom(bg, make([]float32, 64*8)); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	if c.copies.Load() != 1 {
		t.Errorf("copies = %d, want 1", c.copies.Load())
	}
	if c.srcType.Load() != cudasys.MemoryTypeHost || c.dstType.Load() != cudasys.MemoryTypeDevice {
		t.Errorf("HtoD mem types = %d->%d", c.srcType.Load(), c.dstType.Load())
	}
	if c.dstPitch.Load() != 512 || c.widthBytes.Load() != 256 || c.height.Load() != 8 {
		t.Errorf("desc = dstPitch %d width %d height %d, want 512/256/8", c.dstPitch.Load(), c.widthBytes.Load(), c.height.Load())
	}
	if c.dstDevice.Load() != pitchBase {
		t.Errorf("dst device = %#x, want %#x", c.dstDevice.Load(), pitchBase)
	}

	if err := buf.CopyTo(bg, make([]float32, 64*8)); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	if c.srcType.Load() != cudasys.MemoryTypeDevice || c.dstType.Load() != cudasys.MemoryTypeHost {
		t.Errorf("DtoH mem types = %d->%d", c.srcType.Load(), c.dstType.Load())
	}
	if c.srcPitch.Load() != 512 {
		t.Errorf("DtoH src pitch = %d, want 512", c.srcPitch.Load())
	}

	if err := buf.CopyFrom(bg, make([]float32, 10)); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("wrong-length CopyFrom = %v, want ErrLengthMismatch", err)
	}
}

func TestPitchedCopyToDevice(t *testing.T) {
	ctx, c := pitchedFixture(t)
	src, err := AllocPitched[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("alloc src: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })
	dst, err := AllocPitched[float32](ctx, 64, 8)
	if err != nil {
		t.Fatalf("alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	if err := src.CopyToDevice(context.Background(), dst); err != nil {
		t.Fatalf("CopyToDevice: %v", err)
	}
	if c.srcType.Load() != cudasys.MemoryTypeDevice || c.dstType.Load() != cudasys.MemoryTypeDevice {
		t.Errorf("DtoD mem types = %d->%d", c.srcType.Load(), c.dstType.Load())
	}
	if c.srcPitch.Load() != 512 || c.dstPitch.Load() != 512 || c.widthBytes.Load() != 256 {
		t.Errorf("DtoD desc = src %d dst %d width %d", c.srcPitch.Load(), c.dstPitch.Load(), c.widthBytes.Load())
	}

	mismatched, err := AllocPitched[float32](ctx, 32, 8)
	if err != nil {
		t.Fatalf("alloc mismatched: %v", err)
	}
	t.Cleanup(func() { _ = mismatched.Close() })
	if err := src.CopyToDevice(context.Background(), mismatched); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("mismatched dims = %v, want ErrLengthMismatch", err)
	}
}

func TestPitchedSymbolUnavailable(t *testing.T) {
	drv := fakeMemoryDriver(&memCalls{}, pitchBase) // no CuMemAllocPitch bound
	ctx := newTestContext(t, drv)
	if _, err := AllocPitched[float32](ctx, 4, 4); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("err = %v, want ErrSymbolUnavailable", err)
	}
}

func TestPitchedNilReceiver(t *testing.T) {
	var b *PitchedBuffer[float32]
	if b.Width() != 0 || b.Height() != 0 || b.Pitch() != 0 || b.DevicePtr() != 0 {
		t.Error("nil accessors should be zero")
	}
	if err := b.CopyFrom(context.Background(), make([]float32, 4)); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil CopyFrom = %v, want ErrNilBuffer", err)
	}
	if err := b.Close(); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil Close = %v, want ErrNilBuffer", err)
	}
}
