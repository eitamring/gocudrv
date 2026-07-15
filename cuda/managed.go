package cuda

import (
	"context"
	"math"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// memAttachGlobal (CU_MEM_ATTACH_GLOBAL) makes managed memory usable from any
// stream on any device.
const memAttachGlobal uint32 = 1

// deviceCPU is CU_DEVICE_CPU, the prefetch/advise target that means the host.
const deviceCPU cudasys.CUdevice = -1

// MemAdvice is a unified-memory migration hint for ManagedBuffer.Advise. It
// mirrors CUmem_advise.
type MemAdvice int32

const (
	AdviseSetReadMostly          MemAdvice = 1
	AdviseUnsetReadMostly        MemAdvice = 2
	AdviseSetPreferredLocation   MemAdvice = 3
	AdviseUnsetPreferredLocation MemAdvice = 4
	AdviseSetAccessedBy          MemAdvice = 5
	AdviseUnsetAccessedBy        MemAdvice = 6
)

// ManagedBuffer is a region of CUDA unified (managed) memory, addressable from
// both the host and the device. The driver migrates pages between host and
// device on demand; Prefetch and Advise tune that migration. Unlike Buffer, no
// explicit copy is needed: write the host Slice, launch a kernel against
// DevicePtr, and read the Slice back.
//
// Lifetime rule: close a ManagedBuffer before its owning Context, and do not
// touch its Slice after Close.
type ManagedBuffer[T Supported] struct {
	ctx    *Context
	host   *byte
	ptr    cudasys.CUdeviceptr
	length int
	bytes  uint64
	opMu   sync.RWMutex
	closed bool
}

// AllocManaged allocates n elements of unified memory tied to ctx. It returns
// ErrSymbolUnavailable on a driver that does not export cuMemAllocManaged. Close
// the returned buffer before ctx.
func AllocManaged[T Supported](ctx *Context, n int) (*ManagedBuffer[T], error) {
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
	bytes := uint64(n) * es

	var host *byte
	err := ctx.do(context.Background(), func() error {
		p, e := cudaresult.MemAllocManaged(ctx.driver, bytes, memAttachGlobal)
		if e != nil {
			return e
		}
		host = p
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &ManagedBuffer[T]{
		ctx:    ctx,
		host:   host,
		ptr:    cudasys.CUdeviceptr(uintptr(unsafe.Pointer(host))),
		length: n,
		bytes:  bytes,
	}, nil
}

// Len is the element count. It is 0 for a nil receiver.
func (m *ManagedBuffer[T]) Len() int {
	if m == nil {
		return 0
	}
	return m.length
}

// Bytes is the total size in bytes. It is 0 for a nil receiver.
func (m *ManagedBuffer[T]) Bytes() uint64 {
	if m == nil {
		return 0
	}
	return m.bytes
}

// DevicePtr is the device pointer for kernel arguments (pass it with
// ArgDevicePtr). It is 0 for a nil receiver and valid only while the buffer is
// open.
func (m *ManagedBuffer[T]) DevicePtr() cudasys.CUdeviceptr {
	if m == nil {
		return 0
	}
	return m.ptr
}

// Slice returns a host-usable slice over the managed memory; the CPU may read
// and write it directly and the driver migrates pages as needed. It is nil for
// a nil or closed buffer, and is valid only while the buffer is open.
func (m *ManagedBuffer[T]) Slice() []T {
	if m == nil || m.host == nil {
		return nil
	}
	m.opMu.RLock()
	defer m.opMu.RUnlock()
	if m.closed {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(m.host)), m.length)
}

// PrefetchToDevice migrates the buffer to this context's device ahead of a
// kernel launch, ordered on stream, so the first access does not page-fault.
func (m *ManagedBuffer[T]) PrefetchToDevice(ctx context.Context, stream *Stream) error {
	return m.prefetch(ctx, stream, false)
}

// PrefetchToHost migrates the buffer back to the host ahead of CPU access,
// ordered on stream.
func (m *ManagedBuffer[T]) PrefetchToHost(ctx context.Context, stream *Stream) error {
	return m.prefetch(ctx, stream, true)
}

func (m *ManagedBuffer[T]) prefetch(ctx context.Context, stream *Stream, toHost bool) error {
	if m == nil {
		return ErrNilBuffer
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	m.opMu.RLock()
	defer m.opMu.RUnlock()
	if m.closed {
		return ErrBufferClosed
	}
	if stream.ctx != m.ctx {
		return ErrContextMismatch
	}
	dst := m.ctx.device.handle
	if toHost {
		dst = deviceCPU
	}
	ptr, bytes := m.ptr, m.bytes
	return m.ctx.doWait(ctx, func() error {
		return cudaresult.MemPrefetchAsync(m.ctx.driver, ptr, bytes, dst, stream.raw)
	})
}

// Advise applies a unified-memory hint to the whole buffer for this context's
// device (cuMemAdvise), for example AdviseSetReadMostly or
// AdviseSetPreferredLocation.
func (m *ManagedBuffer[T]) Advise(advice MemAdvice) error {
	if m == nil {
		return ErrNilBuffer
	}
	m.opMu.RLock()
	defer m.opMu.RUnlock()
	if m.closed {
		return ErrBufferClosed
	}
	ptr, bytes, dev := m.ptr, m.bytes, m.ctx.device.handle
	return m.ctx.do(context.Background(), func() error {
		return cudaresult.MemAdvise(m.ctx.driver, ptr, bytes, int32(advice), dev)
	})
}

// Close frees the managed allocation with cuMemFree. Idempotent after a
// successful free; a failed free leaves the buffer open to retry. Returns
// ErrContextClosed if the owning Context was closed first.
func (m *ManagedBuffer[T]) Close() error {
	if m == nil {
		return ErrNilBuffer
	}
	m.opMu.Lock()
	defer m.opMu.Unlock()
	if m.closed {
		return nil
	}
	if err := m.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.MemFree(m.ctx.driver, m.ptr)
	}); err != nil {
		return err
	}
	m.closed = true
	return nil
}
