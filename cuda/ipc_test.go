package cuda

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

type ipcCalls struct {
	closes      atomic.Int32
	closedPtr   atomic.Uint64
	openedFlags atomic.Uint32
	eventFlags  atomic.Uint32
}

func ipcFixture(t *testing.T) (*Context, *ipcCalls) {
	t.Helper()
	var c ipcCalls
	drv := fakeMemoryDriver(&memCalls{}, 0x80000)
	drv.CuIpcGetMemHandle = func(h *cudasys.CUipcMemHandle, dptr cudasys.CUdeviceptr) cudasys.CUresult {
		h.Data[0] = byte(dptr)
		h.Data[1] = 0xAB
		return cudasys.CUDA_SUCCESS
	}
	drv.CuIpcOpenMemHandle = func(p *cudasys.CUdeviceptr, h cudasys.CUipcMemHandle, flags uint32) cudasys.CUresult {
		c.openedFlags.Store(flags)
		*p = 0x9000 + cudasys.CUdeviceptr(h.Data[0])
		return cudasys.CUDA_SUCCESS
	}
	drv.CuIpcCloseMemHandle = func(dptr cudasys.CUdeviceptr) cudasys.CUresult {
		c.closes.Add(1)
		c.closedPtr.Store(uint64(dptr))
		return cudasys.CUDA_SUCCESS
	}
	drv.CuEventCreate = func(e *cudasys.CUevent, flags uint32) cudasys.CUresult {
		c.eventFlags.Store(flags)
		*e = 0xE0E0
		return cudasys.CUDA_SUCCESS
	}
	drv.CuEventDestroy = func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuIpcGetEventHandle = func(h *cudasys.CUipcEventHandle, ev cudasys.CUevent) cudasys.CUresult {
		h.Data[0] = 0xEE
		return cudasys.CUDA_SUCCESS
	}
	drv.CuIpcOpenEventHandle = func(ev *cudasys.CUevent, h cudasys.CUipcEventHandle) cudasys.CUresult {
		if h.Data[0] != 0xEE {
			t.Error("event handle not forwarded")
		}
		*ev = 0xFEED
		return cudasys.CUDA_SUCCESS
	}
	return newTestContext(t, drv), &c
}

func TestBufferIPCHandleRoundTrip(t *testing.T) {
	ctx, c := ipcFixture(t)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	h, err := buf.IPCHandle()
	if err != nil {
		t.Fatalf("IPCHandle: %v", err)
	}
	raw := h.Bytes()
	if raw[1] != 0xAB {
		t.Errorf("handle bytes = %v, want driver-filled", raw[:2])
	}
	rebuilt := IPCMemHandleFromBytes(raw)

	imp, err := OpenIPCBuffer[float32](ctx, rebuilt, 16)
	if err != nil {
		t.Fatalf("OpenIPCBuffer: %v", err)
	}
	if c.openedFlags.Load() != cudasys.IpcMemLazyEnablePeerAccess {
		t.Errorf("open flags = %#x, want lazy peer access", c.openedFlags.Load())
	}
	if imp.Len() != 16 || imp.Bytes() != 64 || imp.DevicePtr() == 0 {
		t.Errorf("imported = len %d bytes %d ptr %#x", imp.Len(), imp.Bytes(), imp.DevicePtr())
	}

	bg := context.Background()
	if err := imp.CopyFrom(bg, make([]float32, 16)); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	if err := imp.CopyTo(bg, make([]float32, 16)); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	if err := imp.CopyFrom(bg, make([]float32, 3)); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short CopyFrom = %v, want ErrLengthMismatch", err)
	}

	if err := imp.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := imp.Close(); err != nil {
		t.Errorf("second Close = %v, want nil", err)
	}
	if c.closes.Load() != 1 || c.closedPtr.Load() != uint64(imp.ptr) {
		t.Errorf("closes = %d ptr %#x, want 1 unmap of the imported ptr", c.closes.Load(), c.closedPtr.Load())
	}
	if err := imp.CopyTo(bg, make([]float32, 16)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyTo after close = %v, want ErrBufferClosed", err)
	}
}

func TestIPCRejects(t *testing.T) {
	ctx, _ := ipcFixture(t)
	var nilBuf *Buffer[float32]
	if _, err := nilBuf.IPCHandle(); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil buffer = %v, want ErrNilBuffer", err)
	}
	buf, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	if err := buf.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := buf.IPCHandle(); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("closed buffer = %v, want ErrBufferClosed", err)
	}
	if _, err := OpenIPCBuffer[float32](nil, IPCMemHandle{}, 4); !errors.Is(err, ErrNilContext) {
		t.Errorf("nil ctx = %v, want ErrNilContext", err)
	}
	if _, err := OpenIPCBuffer[float32](ctx, IPCMemHandle{}, 0); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("zero n = %v, want ErrInvalidLength", err)
	}
	ctx.driver.CuIpcOpenMemHandle = nil
	if _, err := OpenIPCBuffer[float32](ctx, IPCMemHandle{}, 4); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing symbol = %v, want ErrSymbolUnavailable", err)
	}
	buf2, err := Alloc[float32](ctx, 4)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf2.Close() })
	ctx.driver.CuIpcGetMemHandle = nil
	if _, err := buf2.IPCHandle(); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("missing get symbol = %v, want ErrSymbolUnavailable", err)
	}
}

func TestEventIPCHandle(t *testing.T) {
	ctx, c := ipcFixture(t)
	ev, err := ctx.NewEvent(WithEventInterprocess())
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	t.Cleanup(func() { _ = ev.Close() })
	if c.eventFlags.Load() != eventInterprocess|eventDisableTiming {
		t.Errorf("create flags = %#x, want interprocess|disableTiming", c.eventFlags.Load())
	}
	h, err := ev.IPCHandle()
	if err != nil {
		t.Fatalf("IPCHandle: %v", err)
	}
	raw := h.Bytes()
	if raw[0] != 0xEE {
		t.Errorf("handle = %v, want driver-filled", raw[:1])
	}

	imported, err := OpenIPCEvent(ctx, IPCEventHandleFromBytes(raw))
	if err != nil {
		t.Fatalf("OpenIPCEvent: %v", err)
	}
	if imported.raw != 0xFEED || !imported.timingDisabled {
		t.Errorf("imported event = %#x timingDisabled %v", imported.raw, imported.timingDisabled)
	}
	if err := imported.Close(); err != nil {
		t.Errorf("imported Close: %v", err)
	}

	plain, err := ctx.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	t.Cleanup(func() { _ = plain.Close() })
	if _, err := plain.IPCHandle(); !errors.Is(err, ErrEventNotInterprocess) {
		t.Errorf("plain event = %v, want ErrEventNotInterprocess", err)
	}
}
