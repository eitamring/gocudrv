package cuda

import (
	"context"
	"errors"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

const (
	memAllocTypePinned   int32  = 1 // CU_MEM_ALLOCATION_TYPE_PINNED
	memLocTypeDevice     int32  = 1 // CU_MEM_LOCATION_TYPE_DEVICE
	memAccessReadWrite   int32  = 3 // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
	granularityRecommend uint32 = 1 // CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
)

// VirtualBuffer is device memory from the virtual memory management API: a
// reserved address range with a physical allocation mapped in. It behaves like a
// Buffer for launches and host copies. Close it before its Context.
type VirtualBuffer[T Supported] struct {
	ctx                       *Context
	ptr                       cudasys.CUdeviceptr
	handle                    cudasys.CUmemGenericAllocationHandle
	size                      uint64 // mapped size, rounded up to the granularity
	length                    int
	bytes                     uint64 // requested element bytes, <= size
	opMu                      sync.RWMutex
	closed                    bool
	mapped, reserved, created bool // live resources, for retryable Close
}

func (c *Context) memProp() cudasys.CUmemAllocationProp {
	return cudasys.CUmemAllocationProp{
		Type:     memAllocTypePinned,
		Location: cudasys.CUmemLocation{Type: memLocTypeDevice, Id: int32(c.device.handle)},
	}
}

// vmmAvailable reports whether every VMM symbol this package uses is bound. All
// eight are checked together so a partial driver cannot produce a buffer that
// can be allocated but not torn down.
func (c *Context) vmmAvailable() bool {
	d := c.driver
	return d.CuMemGetAllocationGranularity != nil && d.CuMemCreate != nil &&
		d.CuMemAddressReserve != nil && d.CuMemMap != nil && d.CuMemSetAccess != nil &&
		d.CuMemUnmap != nil && d.CuMemAddressFree != nil && d.CuMemRelease != nil
}

// tryFree runs one teardown step if its resource is still live, clearing the
// live flag on success and collecting the error otherwise, so Close tears down
// only what remains and can be retried.
func tryFree(live *bool, errs *[]error, op func() error) {
	if !*live {
		return
	}
	if e := op(); e != nil {
		*errs = append(*errs, e)
	} else {
		*live = false
	}
}

// roundUp returns v rounded up to a multiple of mult, and false on uint64
// overflow.
func roundUp(v, mult uint64) (uint64, bool) {
	r := v % mult
	if r == 0 {
		return v, true
	}
	pad := mult - r
	if pad > math.MaxUint64-v {
		return 0, false
	}
	return v + pad, true
}

// acquireVirtual runs the VMM acquisition transaction: create a physical
// allocation, reserve an address range, map it, and grant the device
// read-write access. On any step's failure it rolls back the steps that
// succeeded and returns the original error joined with any rollback errors.
func (c *Context) acquireVirtual(size, gran uint64, prop *cudasys.CUmemAllocationProp) (cudasys.CUdeviceptr, cudasys.CUmemGenericAllocationHandle, error) {
	handle, e := cudaresult.MemCreate(c.driver, size, prop)
	if e != nil {
		return 0, 0, e
	}
	ptr, e := cudaresult.MemAddressReserve(c.driver, size, gran)
	if e != nil {
		return 0, 0, errors.Join(e, cudaresult.MemRelease(c.driver, handle))
	}
	if e := cudaresult.MemMap(c.driver, ptr, size, handle); e != nil {
		return 0, 0, errors.Join(e,
			cudaresult.MemAddressFree(c.driver, ptr, size),
			cudaresult.MemRelease(c.driver, handle))
	}
	desc := cudasys.CUmemAccessDesc{
		Location: cudasys.CUmemLocation{Type: memLocTypeDevice, Id: int32(c.device.handle)},
		Flags:    memAccessReadWrite,
	}
	if e := cudaresult.MemSetAccess(c.driver, ptr, size, &desc); e != nil {
		rb := []error{e, cudaresult.MemUnmap(c.driver, ptr, size)}
		if rb[1] == nil {
			rb = append(rb, cudaresult.MemAddressFree(c.driver, ptr, size))
		}
		return 0, 0, errors.Join(append(rb, cudaresult.MemRelease(c.driver, handle))...)
	}
	return ptr, handle, nil
}

// AllocVirtual allocates n elements of device memory through the VMM API,
// rounding the reservation up to the device's recommended granularity. It
// returns ErrSymbolUnavailable on a driver without the VMM symbols. Close the
// buffer before ctx.
func AllocVirtual[T Supported](ctx *Context, n int) (*VirtualBuffer[T], error) {
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
	if !ctx.vmmAvailable() {
		return nil, ErrSymbolUnavailable
	}

	vb := &VirtualBuffer[T]{ctx: ctx, length: n, bytes: bytes}
	err := ctx.do(context.Background(), func() error {
		prop := ctx.memProp()
		gran, e := cudaresult.MemGetAllocationGranularity(ctx.driver, &prop, granularityRecommend)
		if e != nil {
			return e
		}
		if gran == 0 {
			return ErrInvalidLength
		}
		size, ok := roundUp(bytes, gran)
		if !ok {
			return ErrInvalidLength
		}

		ptr, handle, e := ctx.acquireVirtual(size, gran, &prop)
		if e != nil {
			return e
		}
		vb.ptr, vb.handle, vb.size = ptr, handle, size
		vb.mapped, vb.reserved, vb.created = true, true, true
		return nil
	})
	if err != nil {
		return nil, err
	}
	return vb, nil
}

// Len returns the element count. It is 0 for a nil receiver.
func (b *VirtualBuffer[T]) Len() int {
	if b == nil {
		return 0
	}
	return b.length
}

// Bytes returns the requested size in bytes (the reservation may be larger due
// to granularity rounding). It is 0 for a nil receiver.
func (b *VirtualBuffer[T]) Bytes() uint64 {
	if b == nil {
		return 0
	}
	return b.bytes
}

// DevicePtr is the device pointer for kernel arguments (pass it with
// ArgDevicePtr). It is 0 for a nil receiver and valid only while the buffer is
// open.
func (b *VirtualBuffer[T]) DevicePtr() cudasys.CUdeviceptr {
	if b == nil {
		return 0
	}
	return b.ptr
}

// CopyFrom copies len(src) elements from the host into the buffer. Blocks until
// the copy completes. len(src) must equal Len().
func (b *VirtualBuffer[T]) CopyFrom(ctx context.Context, src []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed || !b.mapped {
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

// CopyTo copies Len() elements from the buffer into the host slice. Blocks
// until the copy completes. len(dst) must equal Len().
func (b *VirtualBuffer[T]) CopyTo(ctx context.Context, dst []T) error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed || !b.mapped {
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

// Close unmaps, releases the handle, and frees the address reservation. It tears
// down only what is still live and is retryable: a partial failure returns an
// error and can be retried. Returns ErrContextClosed if the Context was closed.
func (b *VirtualBuffer[T]) Close() error {
	if b == nil {
		return ErrNilBuffer
	}
	b.opMu.Lock()
	defer b.opMu.Unlock()
	if b.closed {
		return nil
	}
	var errs []error
	outer := b.ctx.doBarrier(context.Background(), func() error {
		tryFree(&b.mapped, &errs, func() error { return cudaresult.MemUnmap(b.ctx.driver, b.ptr, b.size) })
		tryFree(&b.created, &errs, func() error { return cudaresult.MemRelease(b.ctx.driver, b.handle) })
		if !b.mapped {
			tryFree(&b.reserved, &errs, func() error { return cudaresult.MemAddressFree(b.ctx.driver, b.ptr, b.size) })
		}
		return nil
	})
	if outer != nil {
		return outer
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	b.closed = true
	return nil
}
