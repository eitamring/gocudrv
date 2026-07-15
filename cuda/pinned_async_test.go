package cuda

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

var _ PinnedHost[float32] = (*HostBuffer[float32])(nil)
var _ PinnedHost[float32] = (*RegisteredHost[float32])(nil)

type pinnedAsyncCapture struct {
	mu          sync.Mutex
	flatKind    uint8
	flatHost    *byte
	flatDst     cudasys.CUdeviceptr
	flatSrc     cudasys.CUdeviceptr
	flatBytes   uint64
	flatStream  cudasys.CUstream
	two         cudasys.Memcpy2D
	twoStream   cudasys.CUstream
	twoCalls    int
	three       cudasys.Memcpy3D
	threeStream cudasys.CUstream
	threeCalls  int
}

func (c *pinnedAsyncCapture) setFlat(kind uint8, host *byte, dst, src cudasys.CUdeviceptr, bytes uint64, stream cudasys.CUstream) {
	c.mu.Lock()
	c.flatKind = kind
	c.flatHost = host
	c.flatDst = dst
	c.flatSrc = src
	c.flatBytes = bytes
	c.flatStream = stream
	c.mu.Unlock()
}

func (c *pinnedAsyncCapture) flatSnapshot() (uint8, *byte, cudasys.CUdeviceptr, cudasys.CUdeviceptr, uint64, cudasys.CUstream) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.flatKind, c.flatHost, c.flatDst, c.flatSrc, c.flatBytes, c.flatStream
}

func (c *pinnedAsyncCapture) setTwo(desc *cudasys.Memcpy2D, stream cudasys.CUstream) {
	c.mu.Lock()
	c.two = *desc
	c.twoStream = stream
	c.twoCalls++
	c.mu.Unlock()
}

func (c *pinnedAsyncCapture) twoSnapshot() (cudasys.Memcpy2D, cudasys.CUstream, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.two, c.twoStream, c.twoCalls
}

func (c *pinnedAsyncCapture) setThree(desc *cudasys.Memcpy3D, stream cudasys.CUstream) {
	c.mu.Lock()
	c.three = *desc
	c.threeStream = stream
	c.threeCalls++
	c.mu.Unlock()
}

func (c *pinnedAsyncCapture) threeSnapshot() (cudasys.Memcpy3D, cudasys.CUstream, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.three, c.threeStream, c.threeCalls
}

func newPinnedAsyncFixture(t *testing.T) (*Context, *Stream, *pinnedAsyncCapture) {
	t.Helper()
	calls := &pinnedAsyncCapture{}
	drv := fakeMemoryDriver(&memCalls{}, 0x21000)
	var pitchAlloc uint64
	var arrayAlloc uintptr
	drv.CuMemAllocPitch = func(ptr *cudasys.CUdeviceptr, pitch *uint64, _, _ uint64, _ uint32) cudasys.CUresult {
		pitchAlloc++
		*ptr = cudasys.CUdeviceptr(0x30000 + pitchAlloc*0x1000)
		*pitch = 64 + pitchAlloc*32
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemAllocHost = func(ptr **byte, bytes uint64) cudasys.CUresult {
		mem := make([]byte, int(bytes))
		*ptr = &mem[0]
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemFreeHost = func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuMemHostRegister = func(*byte, uint64, uint32) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuMemHostUnregister = func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuStreamCreate = func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
		*stream = 0x5151
		return cudasys.CUDA_SUCCESS
	}
	drv.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuStreamSynchronize = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuArrayCreate = func(array *cudasys.CUarray, _ *cudasys.CUDA_ARRAY_DESCRIPTOR) cudasys.CUresult {
		arrayAlloc++
		*array = cudasys.CUarray(0xA000 + arrayAlloc)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuArrayDestroy = func(cudasys.CUarray) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuMemcpyHtoD = func(dst cudasys.CUdeviceptr, host *byte, bytes uint64) cudasys.CUresult {
		calls.setFlat(1, host, dst, 0, bytes, 0)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpyDtoH = func(host *byte, src cudasys.CUdeviceptr, bytes uint64) cudasys.CUresult {
		calls.setFlat(2, host, 0, src, bytes, 0)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpyHtoDAsync = func(dst cudasys.CUdeviceptr, host *byte, bytes uint64, stream cudasys.CUstream) cudasys.CUresult {
		calls.setFlat(3, host, dst, 0, bytes, stream)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpyDtoHAsync = func(host *byte, src cudasys.CUdeviceptr, bytes uint64, stream cudasys.CUstream) cudasys.CUresult {
		calls.setFlat(4, host, 0, src, bytes, stream)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpy2DAsync = func(desc *cudasys.Memcpy2D, stream cudasys.CUstream) cudasys.CUresult {
		calls.setTwo(desc, stream)
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemcpy3DAsync = func(desc *cudasys.Memcpy3D, stream cudasys.CUstream) cudasys.CUresult {
		calls.setThree(desc, stream)
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, drv)
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	return ctx, stream, calls
}

func assertFlatCopy(t *testing.T, calls *pinnedAsyncCapture, kind uint8, host *byte, device cudasys.CUdeviceptr, stream cudasys.CUstream) {
	t.Helper()
	gotKind, gotHost, dst, src, bytes, gotStream := calls.flatSnapshot()
	if gotKind != kind || gotHost != host || bytes != 16 || gotStream != stream {
		t.Fatalf("flat copy = kind %d host %p bytes %d stream %#x, want %d %p 16 %#x", gotKind, gotHost, bytes, gotStream, kind, host, stream)
	}
	if kind == 1 || kind == 3 {
		if dst != device {
			t.Fatalf("flat destination = %#x, want %#x", dst, device)
		}
	} else if src != device {
		t.Fatalf("flat source = %#x, want %#x", src, device)
	}
}

func TestPinnedHostFlatCopies(t *testing.T) {
	ctx, stream, calls := newPinnedAsyncFixture(t)
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	host, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = host.Close() })
	mem := make([]float32, 4)
	registered, err := RegisterHost(ctx, mem)
	if err != nil {
		t.Fatalf("RegisterHost: %v", err)
	}
	t.Cleanup(func() { _ = registered.Close() })

	cases := []struct {
		name string
		host PinnedHost[float32]
		ptr  *byte
	}{
		{name: "allocated", host: host, ptr: host.ptr},
		{name: "registered", host: registered, ptr: registered.ptr},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := buf.CopyFromHost(context.Background(), tc.host); err != nil {
				t.Fatalf("CopyFromHost: %v", err)
			}
			assertFlatCopy(t, calls, 1, tc.ptr, buf.ptr, 0)
			if err := buf.CopyToHost(context.Background(), tc.host); err != nil {
				t.Fatalf("CopyToHost: %v", err)
			}
			assertFlatCopy(t, calls, 2, tc.ptr, buf.ptr, 0)
			if err := buf.CopyFromHostAsync(context.Background(), stream, tc.host); err != nil {
				t.Fatalf("CopyFromHostAsync: %v", err)
			}
			assertFlatCopy(t, calls, 3, tc.ptr, buf.ptr, stream.raw)
			if err := buf.CopyToHostAsync(context.Background(), stream, tc.host); err != nil {
				t.Fatalf("CopyToHostAsync: %v", err)
			}
			assertFlatCopy(t, calls, 4, tc.ptr, buf.ptr, stream.raw)
		})
	}
}

func TestPinnedAsyncDescriptors(t *testing.T) {
	ctx, stream, calls := newPinnedAsyncFixture(t)
	host6, err := AllocHost[float32](ctx, 6)
	if err != nil {
		t.Fatalf("AllocHost 6: %v", err)
	}
	t.Cleanup(func() { _ = host6.Close() })
	registered6, err := RegisterHost(ctx, make([]float32, 6))
	if err != nil {
		t.Fatalf("RegisterHost 6: %v", err)
	}
	t.Cleanup(func() { _ = registered6.Close() })
	src, err := AllocPitched[float32](ctx, 3, 2)
	if err != nil {
		t.Fatalf("AllocPitched source: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })
	dst, err := AllocPitched[float32](ctx, 3, 2)
	if err != nil {
		t.Fatalf("AllocPitched destination: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	if err := src.CopyFromHostAsync(context.Background(), stream, host6); err != nil {
		t.Fatalf("pitched CopyFromHostAsync: %v", err)
	}
	desc2, gotStream, _ := calls.twoSnapshot()
	if desc2.SrcMemoryType != cudasys.MemoryTypeHost || desc2.SrcHost != unsafe.Pointer(host6.ptr) || desc2.SrcPitch != 12 {
		t.Fatalf("pitched host source = type %d ptr %p pitch %d", desc2.SrcMemoryType, desc2.SrcHost, desc2.SrcPitch)
	}
	if desc2.DstMemoryType != cudasys.MemoryTypeDevice || desc2.DstDevice != src.ptr || desc2.DstPitch != src.pitch {
		t.Fatalf("pitched device destination = type %d ptr %#x pitch %d", desc2.DstMemoryType, desc2.DstDevice, desc2.DstPitch)
	}
	if desc2.WidthInBytes != 12 || desc2.Height != 2 || gotStream != stream.raw {
		t.Fatalf("pitched geometry = %dx%d stream %#x", desc2.WidthInBytes, desc2.Height, gotStream)
	}

	if err := src.CopyToHostAsync(context.Background(), stream, registered6); err != nil {
		t.Fatalf("pitched CopyToHostAsync: %v", err)
	}
	desc2, gotStream, _ = calls.twoSnapshot()
	if desc2.SrcMemoryType != cudasys.MemoryTypeDevice || desc2.SrcDevice != src.ptr || desc2.SrcPitch != src.pitch {
		t.Fatalf("pitched device source = type %d ptr %#x pitch %d", desc2.SrcMemoryType, desc2.SrcDevice, desc2.SrcPitch)
	}
	if desc2.DstMemoryType != cudasys.MemoryTypeHost || desc2.DstHost != unsafe.Pointer(registered6.ptr) || desc2.DstPitch != 12 {
		t.Fatalf("pitched host destination = type %d ptr %p pitch %d", desc2.DstMemoryType, desc2.DstHost, desc2.DstPitch)
	}
	if desc2.WidthInBytes != 12 || desc2.Height != 2 || gotStream != stream.raw {
		t.Fatalf("pitched reverse geometry = %dx%d stream %#x", desc2.WidthInBytes, desc2.Height, gotStream)
	}

	if err := src.CopyToDeviceAsync(context.Background(), stream, dst); err != nil {
		t.Fatalf("pitched CopyToDeviceAsync: %v", err)
	}
	desc2, gotStream, _ = calls.twoSnapshot()
	if desc2.SrcMemoryType != cudasys.MemoryTypeDevice || desc2.SrcDevice != src.ptr || desc2.SrcPitch != src.pitch {
		t.Fatalf("DtoD source = type %d ptr %#x pitch %d", desc2.SrcMemoryType, desc2.SrcDevice, desc2.SrcPitch)
	}
	if desc2.DstMemoryType != cudasys.MemoryTypeDevice || desc2.DstDevice != dst.ptr || desc2.DstPitch != dst.pitch {
		t.Fatalf("DtoD destination = type %d ptr %#x pitch %d", desc2.DstMemoryType, desc2.DstDevice, desc2.DstPitch)
	}
	if desc2.WidthInBytes != 12 || desc2.Height != 2 || gotStream != stream.raw || src.pitch == dst.pitch {
		t.Fatalf("DtoD geometry = %dx%d pitches %d/%d stream %#x", desc2.WidthInBytes, desc2.Height, src.pitch, dst.pitch, gotStream)
	}
	if err := src.CopyToDeviceAsync(context.Background(), stream, src); err != nil {
		t.Fatalf("pitched same-buffer copy: %v", err)
	}
	desc2, _, _ = calls.twoSnapshot()
	if desc2.SrcDevice != src.ptr || desc2.DstDevice != src.ptr || desc2.SrcPitch != src.pitch || desc2.DstPitch != src.pitch {
		t.Fatalf("same-buffer descriptor = %+v", desc2)
	}

	host12, err := AllocHost[float32](ctx, 12)
	if err != nil {
		t.Fatalf("AllocHost 12: %v", err)
	}
	t.Cleanup(func() { _ = host12.Close() })
	registered12, err := RegisterHost(ctx, make([]float32, 12))
	if err != nil {
		t.Fatalf("RegisterHost 12: %v", err)
	}
	t.Cleanup(func() { _ = registered12.Close() })
	volume, err := AllocVolume[float32](ctx, 3, 2, 2)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = volume.Close() })
	if err := volume.CopyFromHostAsync(context.Background(), stream, registered12); err != nil {
		t.Fatalf("volume CopyFromHostAsync: %v", err)
	}
	desc3, gotStream3, _ := calls.threeSnapshot()
	if desc3.SrcMemoryType != cudasys.MemoryTypeHost || desc3.SrcHost != unsafe.Pointer(registered12.ptr) || desc3.SrcPitch != 12 || desc3.SrcHeight != 2 {
		t.Fatalf("volume host source = type %d ptr %p pitch %d height %d", desc3.SrcMemoryType, desc3.SrcHost, desc3.SrcPitch, desc3.SrcHeight)
	}
	if desc3.DstMemoryType != cudasys.MemoryTypeDevice || desc3.DstDevice != volume.ptr || desc3.DstPitch != volume.pitch || desc3.DstHeight != 2 {
		t.Fatalf("volume device destination = type %d ptr %#x pitch %d height %d", desc3.DstMemoryType, desc3.DstDevice, desc3.DstPitch, desc3.DstHeight)
	}
	if desc3.WidthInBytes != 12 || desc3.Height != 2 || desc3.Depth != 2 || gotStream3 != stream.raw {
		t.Fatalf("volume geometry = %dx%dx%d stream %#x", desc3.WidthInBytes, desc3.Height, desc3.Depth, gotStream3)
	}
	if err := volume.CopyToHostAsync(context.Background(), stream, host12); err != nil {
		t.Fatalf("volume CopyToHostAsync: %v", err)
	}
	desc3, gotStream3, _ = calls.threeSnapshot()
	if desc3.SrcMemoryType != cudasys.MemoryTypeDevice || desc3.SrcDevice != volume.ptr || desc3.SrcPitch != volume.pitch || desc3.SrcHeight != 2 {
		t.Fatalf("volume device source = type %d ptr %#x pitch %d height %d", desc3.SrcMemoryType, desc3.SrcDevice, desc3.SrcPitch, desc3.SrcHeight)
	}
	if desc3.DstMemoryType != cudasys.MemoryTypeHost || desc3.DstHost != unsafe.Pointer(host12.ptr) || desc3.DstPitch != 12 || desc3.DstHeight != 2 || gotStream3 != stream.raw {
		t.Fatalf("volume host destination = type %d ptr %p pitch %d height %d stream %#x", desc3.DstMemoryType, desc3.DstHost, desc3.DstPitch, desc3.DstHeight, gotStream3)
	}

	array, err := AllocArray2D[float32](ctx, 3, 2)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = array.Close() })
	if err := array.CopyFromHostAsync(context.Background(), stream, host6); err != nil {
		t.Fatalf("array CopyFromHostAsync: %v", err)
	}
	desc2, gotStream, _ = calls.twoSnapshot()
	if desc2.SrcMemoryType != cudasys.MemoryTypeHost || desc2.SrcHost != unsafe.Pointer(host6.ptr) || desc2.SrcPitch != 12 {
		t.Fatalf("array host source = type %d ptr %p pitch %d", desc2.SrcMemoryType, desc2.SrcHost, desc2.SrcPitch)
	}
	if desc2.DstMemoryType != cudasys.MemoryTypeArray || desc2.DstArray != uintptr(array.handle) || desc2.DstPitch != 0 || desc2.WidthInBytes != 12 || desc2.Height != 2 || gotStream != stream.raw {
		t.Fatalf("array destination = type %d handle %#x pitch %d geometry %dx%d stream %#x", desc2.DstMemoryType, desc2.DstArray, desc2.DstPitch, desc2.WidthInBytes, desc2.Height, gotStream)
	}
	if err := array.CopyToHostAsync(context.Background(), stream, registered6); err != nil {
		t.Fatalf("array CopyToHostAsync: %v", err)
	}
	desc2, gotStream, _ = calls.twoSnapshot()
	if desc2.SrcMemoryType != cudasys.MemoryTypeArray || desc2.SrcArray != uintptr(array.handle) || desc2.SrcPitch != 0 {
		t.Fatalf("array source = type %d handle %#x pitch %d", desc2.SrcMemoryType, desc2.SrcArray, desc2.SrcPitch)
	}
	if desc2.DstMemoryType != cudasys.MemoryTypeHost || desc2.DstHost != unsafe.Pointer(registered6.ptr) || desc2.DstPitch != 12 || desc2.WidthInBytes != 12 || desc2.Height != 2 || gotStream != stream.raw {
		t.Fatalf("array host destination = type %d ptr %p pitch %d geometry %dx%d stream %#x", desc2.DstMemoryType, desc2.DstHost, desc2.DstPitch, desc2.WidthInBytes, desc2.Height, gotStream)
	}
}

func TestPinnedAsyncTypedNilHosts(t *testing.T) {
	ctx, stream, _ := newPinnedAsyncFixture(t)
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	pitched, err := AllocPitched[float32](ctx, 2, 2)
	if err != nil {
		t.Fatalf("AllocPitched: %v", err)
	}
	t.Cleanup(func() { _ = pitched.Close() })
	volume, err := AllocVolume[float32](ctx, 2, 2, 1)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = volume.Close() })
	array, err := AllocArray2D[float32](ctx, 2, 2)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = array.Close() })
	operations := []struct {
		name string
		call func(PinnedHost[float32]) error
	}{
		{name: "buffer from", call: func(host PinnedHost[float32]) error { return buf.CopyFromHost(context.Background(), host) }},
		{name: "buffer to", call: func(host PinnedHost[float32]) error { return buf.CopyToHost(context.Background(), host) }},
		{name: "buffer from async", call: func(host PinnedHost[float32]) error { return buf.CopyFromHostAsync(context.Background(), stream, host) }},
		{name: "buffer to async", call: func(host PinnedHost[float32]) error { return buf.CopyToHostAsync(context.Background(), stream, host) }},
		{name: "pitched from", call: func(host PinnedHost[float32]) error {
			return pitched.CopyFromHostAsync(context.Background(), stream, host)
		}},
		{name: "pitched to", call: func(host PinnedHost[float32]) error {
			return pitched.CopyToHostAsync(context.Background(), stream, host)
		}},
		{name: "volume from", call: func(host PinnedHost[float32]) error {
			return volume.CopyFromHostAsync(context.Background(), stream, host)
		}},
		{name: "volume to", call: func(host PinnedHost[float32]) error {
			return volume.CopyToHostAsync(context.Background(), stream, host)
		}},
		{name: "array from", call: func(host PinnedHost[float32]) error {
			return array.CopyFromHostAsync(context.Background(), stream, host)
		}},
		{name: "array to", call: func(host PinnedHost[float32]) error { return array.CopyToHostAsync(context.Background(), stream, host) }},
	}
	var nilInterface PinnedHost[float32]
	var nilAllocated *HostBuffer[float32]
	var nilRegistered *RegisteredHost[float32]
	hosts := []struct {
		name string
		host PinnedHost[float32]
	}{
		{name: "nil interface", host: nilInterface},
		{name: "nil allocated", host: nilAllocated},
		{name: "nil registered", host: nilRegistered},
	}
	for _, operation := range operations {
		for _, host := range hosts {
			t.Run(operation.name+"/"+host.name, func(t *testing.T) {
				if err := operation.call(host.host); !errors.Is(err, ErrNilBuffer) {
					t.Fatalf("error = %v, want ErrNilBuffer", err)
				}
			})
		}
	}
	var nilPitched *PitchedBuffer[float32]
	if err := nilPitched.CopyFromHostAsync(context.Background(), stream, nilInterface); !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil pitched error = %v, want ErrNilBuffer", err)
	}
	var nilVolume *Volume[float32]
	if err := nilVolume.CopyFromHostAsync(context.Background(), stream, nilInterface); !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil volume error = %v, want ErrNilBuffer", err)
	}
	var nilArray *Array2D[float32]
	if err := nilArray.CopyFromHostAsync(context.Background(), stream, nilInterface); !errors.Is(err, ErrNilArray) {
		t.Fatalf("nil array error = %v, want ErrNilArray", err)
	}
}

func TestPinnedAsyncValidationAndDispatchErrors(t *testing.T) {
	ctx, stream, calls := newPinnedAsyncFixture(t)
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	pitched, err := AllocPitched[float32](ctx, 2, 2)
	if err != nil {
		t.Fatalf("AllocPitched: %v", err)
	}
	t.Cleanup(func() { _ = pitched.Close() })
	volume, err := AllocVolume[float32](ctx, 2, 2, 1)
	if err != nil {
		t.Fatalf("AllocVolume: %v", err)
	}
	t.Cleanup(func() { _ = volume.Close() })
	array, err := AllocArray2D[float32](ctx, 2, 2)
	if err != nil {
		t.Fatalf("AllocArray2D: %v", err)
	}
	t.Cleanup(func() { _ = array.Close() })
	valid, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost valid: %v", err)
	}
	t.Cleanup(func() { _ = valid.Close() })
	short, err := AllocHost[float32](ctx, 3)
	if err != nil {
		t.Fatalf("AllocHost short: %v", err)
	}
	t.Cleanup(func() { _ = short.Close() })
	closed, err := AllocHost[float32](ctx, 4)
	if err != nil {
		t.Fatalf("AllocHost closed: %v", err)
	}
	if err := closed.Close(); err != nil {
		t.Fatalf("close host: %v", err)
	}
	closedRegistered, err := RegisterHost(ctx, make([]float32, 4))
	if err != nil {
		t.Fatalf("RegisterHost closed: %v", err)
	}
	if err := closedRegistered.Close(); err != nil {
		t.Fatalf("close registration: %v", err)
	}
	foreign := &HostBuffer[float32]{ctx: &Context{}, ptr: valid.ptr, length: 4, bytes: 16}

	hostOperations := []struct {
		name string
		call func(PinnedHost[float32]) error
	}{
		{name: "buffer", call: func(host PinnedHost[float32]) error { return buf.CopyFromHostAsync(context.Background(), stream, host) }},
		{name: "pitched", call: func(host PinnedHost[float32]) error {
			return pitched.CopyFromHostAsync(context.Background(), stream, host)
		}},
		{name: "volume", call: func(host PinnedHost[float32]) error {
			return volume.CopyFromHostAsync(context.Background(), stream, host)
		}},
		{name: "array", call: func(host PinnedHost[float32]) error {
			return array.CopyFromHostAsync(context.Background(), stream, host)
		}},
	}
	badHosts := []struct {
		name string
		host PinnedHost[float32]
		err  error
	}{
		{name: "closed allocated", host: closed, err: ErrBufferClosed},
		{name: "closed registered", host: closedRegistered, err: ErrBufferClosed},
		{name: "foreign", host: foreign, err: ErrContextMismatch},
		{name: "short", host: short, err: ErrLengthMismatch},
	}
	for _, operation := range hostOperations {
		for _, bad := range badHosts {
			t.Run(operation.name+"/"+bad.name, func(t *testing.T) {
				if err := operation.call(bad.host); !errors.Is(err, bad.err) {
					t.Fatalf("error = %v, want %v", err, bad.err)
				}
			})
		}
	}
	if err := buf.CopyFromHost(context.Background(), foreign); !errors.Is(err, ErrContextMismatch) {
		t.Fatalf("sync foreign source = %v, want ErrContextMismatch", err)
	}
	if err := buf.CopyToHost(context.Background(), foreign); !errors.Is(err, ErrContextMismatch) {
		t.Fatalf("sync foreign destination = %v, want ErrContextMismatch", err)
	}
	closedStream := &Stream{ctx: ctx, raw: 0x6161, closed: true}
	if err := pitched.CopyFromHostAsync(context.Background(), closedStream, valid); !errors.Is(err, ErrStreamClosed) {
		t.Fatalf("closed stream = %v, want ErrStreamClosed", err)
	}
	if err := pitched.CopyFromHostAsync(context.Background(), nil, valid); !errors.Is(err, ErrNilStream) {
		t.Fatalf("nil stream = %v, want ErrNilStream", err)
	}
	foreignStream := &Stream{ctx: &Context{}, raw: 0x7171}
	if err := pitched.CopyFromHostAsync(context.Background(), foreignStream, valid); !errors.Is(err, ErrContextMismatch) {
		t.Fatalf("foreign stream = %v, want ErrContextMismatch", err)
	}

	buf.closed = true
	if err := buf.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrBufferClosed) {
		t.Fatalf("closed buffer = %v, want ErrBufferClosed", err)
	}
	buf.closed = false
	pitched.closed = true
	if err := pitched.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrBufferClosed) {
		t.Fatalf("closed pitched = %v, want ErrBufferClosed", err)
	}
	pitched.closed = false
	volume.closed = true
	if err := volume.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrBufferClosed) {
		t.Fatalf("closed volume = %v, want ErrBufferClosed", err)
	}
	volume.closed = false
	array.closed = true
	if err := array.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrArrayClosed) {
		t.Fatalf("closed array = %v, want ErrArrayClosed", err)
	}
	array.closed = false

	copy2 := ctx.driver.CuMemcpy2DAsync
	ctx.driver.CuMemcpy2DAsync = nil
	if err := pitched.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrSymbolUnavailable) {
		t.Fatalf("missing 2D symbol = %v, want ErrSymbolUnavailable", err)
	}
	if err := array.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrSymbolUnavailable) {
		t.Fatalf("missing array copy symbol = %v, want ErrSymbolUnavailable", err)
	}
	ctx.driver.CuMemcpy2DAsync = copy2
	copy3 := ctx.driver.CuMemcpy3DAsync
	ctx.driver.CuMemcpy3DAsync = nil
	if err := volume.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrSymbolUnavailable) {
		t.Fatalf("missing 3D symbol = %v, want ErrSymbolUnavailable", err)
	}
	ctx.driver.CuMemcpy3DAsync = copy3
	flat := ctx.driver.CuMemcpyHtoDAsync
	ctx.driver.CuMemcpyHtoDAsync = nil
	if err := buf.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("missing flat symbol = %v, want ErrNotInitialized", err)
	}
	ctx.driver.CuMemcpyHtoDAsync = flat

	ctx.driver.CuMemcpy2DAsync = func(*cudasys.Memcpy2D, cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_VALUE }
	if err := pitched.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("2D driver error = %v, want ErrInvalidValue", err)
	}
	ctx.driver.CuMemcpy2DAsync = copy2
	ctx.driver.CuMemcpy3DAsync = func(*cudasys.Memcpy3D, cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_VALUE }
	if err := volume.CopyFromHostAsync(context.Background(), stream, valid); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("3D driver error = %v, want ErrInvalidValue", err)
	}
	ctx.driver.CuMemcpy3DAsync = copy3

	_, _, before2 := calls.twoSnapshot()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := pitched.CopyFromHostAsync(canceled, stream, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled 2D = %v, want context.Canceled", err)
	}
	_, _, after2 := calls.twoSnapshot()
	if after2 != before2 {
		t.Fatalf("canceled 2D reached driver: before %d after %d", before2, after2)
	}
	_, _, before3 := calls.threeSnapshot()
	if err := volume.CopyFromHostAsync(canceled, stream, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled 3D = %v, want context.Canceled", err)
	}
	_, _, after3 := calls.threeSnapshot()
	if after3 != before3 {
		t.Fatalf("canceled 3D reached driver: before %d after %d", before3, after3)
	}
}

func TestPinnedAsyncOpsReset(t *testing.T) {
	var value byte
	driver := &cudasys.Driver{}
	two := &memcpy2DAsyncOp{
		driver: driver,
		desc: cudasys.Memcpy2D{
			SrcHost:      unsafe.Pointer(&value),
			DstHost:      unsafe.Pointer(&value),
			WidthInBytes: 16,
			Height:       2,
		},
		stream: 0x5151,
	}
	two.reset()
	if two.driver != nil || two.stream != 0 || two.desc != (cudasys.Memcpy2D{}) {
		t.Fatalf("2D async op retained state: %+v", two)
	}
	three := &memcpy3DAsyncOp{
		driver: driver,
		desc: cudasys.Memcpy3D{
			SrcHost:      unsafe.Pointer(&value),
			DstHost:      unsafe.Pointer(&value),
			WidthInBytes: 16,
			Height:       2,
			Depth:        3,
		},
		stream: 0x5151,
	}
	three.reset()
	if three.driver != nil || three.stream != 0 || three.desc != (cudasys.Memcpy3D{}) {
		t.Fatalf("3D async op retained state: %+v", three)
	}
}

func TestPitchedAsyncDeviceCopyUsesStableLockOrder(t *testing.T) {
	ctx, stream, _ := newPinnedAsyncFixture(t)
	a, err := AllocPitched[float32](ctx, 2, 2)
	if err != nil {
		t.Fatalf("AllocPitched a: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := AllocPitched[float32](ctx, 2, 2)
	if err != nil {
		t.Fatalf("AllocPitched b: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	low, high := a, b
	if uintptr(unsafe.Pointer(low)) > uintptr(unsafe.Pointer(high)) {
		low, high = high, low
	}
	low.opMu.Lock()
	locked := true
	defer func() {
		if locked {
			low.opMu.Unlock()
		}
	}()
	stream.opMu.Lock()
	started := make(chan struct{})
	copyDone := make(chan error, 1)
	go func() {
		close(started)
		copyDone <- high.CopyToDeviceAsync(context.Background(), stream, low)
	}()
	<-started
	stream.opMu.Unlock()
	deadline := time.Now().Add(2 * time.Second)
	for stream.opMu.TryLock() {
		stream.opMu.Unlock()
		if time.Now().After(deadline) {
			low.opMu.Unlock()
			locked = false
			<-copyDone
			t.Fatal("copy did not acquire the stream lock")
		}
		runtime.Gosched()
	}
	if !high.opMu.TryLock() {
		low.opMu.Unlock()
		locked = false
		<-copyDone
		t.Fatal("copy locked the higher-address buffer before the lower-address buffer")
	}
	high.opMu.Unlock()
	low.opMu.Unlock()
	locked = false
	if err := <-copyDone; err != nil {
		t.Fatalf("CopyToDeviceAsync: %v", err)
	}
}

func TestPinnedAsyncHoldsRegisteredHostLockThroughSubmission(t *testing.T) {
	ctx, stream, _ := newPinnedAsyncFixture(t)
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	registered, err := RegisterHost(ctx, make([]float32, 4))
	if err != nil {
		t.Fatalf("RegisterHost: %v", err)
	}
	t.Cleanup(func() { _ = registered.Close() })
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() {
		releaseOnce.Do(func() { close(release) })
	}
	defer unblock()
	ctx.driver.CuMemcpyHtoDAsync = func(cudasys.CUdeviceptr, *byte, uint64, cudasys.CUstream) cudasys.CUresult {
		close(entered)
		<-release
		return cudasys.CUDA_SUCCESS
	}
	copyDone := make(chan error, 1)
	go func() {
		copyDone <- buf.CopyFromHostAsync(context.Background(), stream, registered)
	}()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("copy did not reach the driver")
	}
	closeDone := make(chan error, 1)
	go func() {
		closeDone <- registered.Close()
	}()
	select {
	case err := <-closeDone:
		t.Fatalf("Close returned during submission: %v", err)
	case <-time.After(20 * time.Millisecond):
	}
	unblock()
	if err := <-copyDone; err != nil {
		t.Fatalf("copy: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Fatalf("Close: %v", err)
	}
}
