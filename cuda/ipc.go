package cuda

import (
	"context"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// IPCMemHandle is an opaque 64-byte token another process uses to map this
// process's device allocation. Ship Bytes over any channel (pipe, socket, file)
// and rebuild it with IPCMemHandleFromBytes.
type IPCMemHandle struct {
	h cudasys.CUipcMemHandle
}

// Bytes returns the handle's raw 64 bytes.
func (h IPCMemHandle) Bytes() [64]byte { return h.h.Data }

// IPCMemHandleFromBytes rebuilds a handle received from another process.
func IPCMemHandleFromBytes(b [64]byte) IPCMemHandle {
	return IPCMemHandle{h: cudasys.CUipcMemHandle{Data: b}}
}

// IPCEventHandle is the event counterpart of IPCMemHandle.
type IPCEventHandle struct {
	h cudasys.CUipcEventHandle
}

// Bytes returns the handle's raw 64 bytes.
func (h IPCEventHandle) Bytes() [64]byte { return h.h.Data }

// IPCEventHandleFromBytes rebuilds a handle received from another process.
func IPCEventHandleFromBytes(b [64]byte) IPCEventHandle {
	return IPCEventHandle{h: cudasys.CUipcEventHandle{Data: b}}
}

// IPCHandle exports the buffer for another process (cuIpcGetMemHandle). The
// buffer must stay open while any process has it mapped. Returns
// ErrSymbolUnavailable on a driver without the IPC symbols.
func (b *Buffer[T]) IPCHandle() (IPCMemHandle, error) {
	if b == nil {
		return IPCMemHandle{}, ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return IPCMemHandle{}, ErrBufferClosed
	}
	var h cudasys.CUipcMemHandle
	err := b.ctx.doWait(context.Background(), func() error {
		got, e := cudaresult.IpcGetMemHandle(b.ctx.driver, b.ptr)
		if e != nil {
			return e
		}
		h = got
		return nil
	})
	if err != nil {
		return IPCMemHandle{}, err
	}
	return IPCMemHandle{h: h}, nil
}

// IPCBuffer is another process's device allocation mapped into this process
// (cuIpcOpenMemHandle). It copies like a Buffer; Close unmaps the handle here
// and never frees the exporter's memory. Close it before the Context.
type IPCBuffer[T Supported] struct {
	ctx    *Context
	ptr    cudasys.CUdeviceptr
	length int
	bytes  uint64
	opMu   sync.RWMutex
	closed bool
}

// OpenIPCBuffer maps a handle exported by another process as n elements of T
// (peer access is enabled lazily). The element count is not carried by the
// handle, so n must match what the exporter allocated. Opening a handle in the
// process that exported it fails with ErrInvalidContext.
func OpenIPCBuffer[T Supported](ctx *Context, h IPCMemHandle, n int) (*IPCBuffer[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if n <= 0 {
		return nil, ErrInvalidLength
	}
	es := elemSize[T]()
	if uint64(n) > math.MaxUint64/es {
		return nil, ErrInvalidLength
	}

	var ptr cudasys.CUdeviceptr
	err := ctx.do(context.Background(), func() error {
		p, e := cudaresult.IpcOpenMemHandle(ctx.driver, h.h, cudasys.IpcMemLazyEnablePeerAccess)
		if e != nil {
			return e
		}
		ptr = p
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &IPCBuffer[T]{ctx: ctx, ptr: ptr, length: n, bytes: uint64(n) * es}, nil
}

// Len returns the element count. It is 0 for a nil receiver.
func (b *IPCBuffer[T]) Len() int {
	if b == nil {
		return 0
	}
	return b.length
}

// Bytes returns the mapped size in bytes. It is 0 for a nil receiver.
func (b *IPCBuffer[T]) Bytes() uint64 {
	if b == nil {
		return 0
	}
	return b.bytes
}

// DevicePtr is the mapped device pointer for kernel arguments (pass it with
// ArgDevicePtr). It is 0 for a nil receiver and valid only while open.
func (b *IPCBuffer[T]) DevicePtr() cudasys.CUdeviceptr {
	if b == nil {
		return 0
	}
	return b.ptr
}

// CopyFrom copies len(src) elements from the host into the mapped memory.
// Blocks until the copy completes. len(src) must equal Len().
func (b *IPCBuffer[T]) CopyFrom(ctx context.Context, src []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if len(src) == 0 || len(src) != b.length {
		return ErrLengthMismatch
	}
	srcPtr := (*byte)(unsafe.Pointer(&src[0]))
	err := b.ctx.memcpyHtoD(ctx, b.ptr, srcPtr, b.bytes)
	runtime.KeepAlive(src)
	return err
}

// CopyTo copies Len() elements from the mapped memory into the host slice.
// Blocks until the copy completes. len(dst) must equal Len().
func (b *IPCBuffer[T]) CopyTo(ctx context.Context, dst []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return ErrBufferClosed
	}
	if len(dst) == 0 || len(dst) != b.length {
		return ErrLengthMismatch
	}
	dstPtr := (*byte)(unsafe.Pointer(&dst[0]))
	err := b.ctx.memcpyDtoH(ctx, dstPtr, b.ptr, b.bytes)
	runtime.KeepAlive(dst)
	return err
}

// Close unmaps the handle in this process (cuIpcCloseMemHandle); the
// exporter's allocation is untouched. Idempotent; a failed unmap leaves the
// buffer open to retry.
func (b *IPCBuffer[T]) Close() error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.Lock()
	defer b.opMu.Unlock()
	if b.closed {
		return nil
	}
	if err := b.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.IpcCloseMemHandle(b.ctx.driver, b.ptr)
	}); err != nil {
		return err
	}
	b.closed = true
	return nil
}

// IPCHandle exports the event for another process (cuIpcGetEventHandle). The
// event must have been created with WithEventInterprocess (else
// ErrEventNotInterprocess) and must stay open while any imported reference to
// it is in use.
func (e *Event) IPCHandle() (IPCEventHandle, error) {
	if e == nil {
		return IPCEventHandle{}, ErrNilEvent
	}
	e.opMu.RLock()
	defer e.opMu.RUnlock()
	if e.closed {
		return IPCEventHandle{}, ErrEventClosed
	}
	if !e.interprocess {
		return IPCEventHandle{}, ErrEventNotInterprocess
	}
	var h cudasys.CUipcEventHandle
	err := e.ctx.doWait(context.Background(), func() error {
		got, err := cudaresult.IpcGetEventHandle(e.ctx.driver, e.raw)
		if err != nil {
			return err
		}
		h = got
		return nil
	})
	if err != nil {
		return IPCEventHandle{}, err
	}
	return IPCEventHandle{h: h}, nil
}

// OpenIPCEvent imports an event exported by another process
// (cuIpcOpenEventHandle). The result behaves like a timing-disabled Event;
// Close releases this process's reference.
func OpenIPCEvent(ctx *Context, h IPCEventHandle) (*Event, error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	var raw cudasys.CUevent
	err := ctx.do(context.Background(), func() error {
		ev, e := cudaresult.IpcOpenEventHandle(ctx.driver, h.h)
		if e != nil {
			return e
		}
		raw = ev
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Event{ctx: ctx, raw: raw, timingDisabled: true, interprocess: true}, nil
}
