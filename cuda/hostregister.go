package cuda

import (
	"context"
	"runtime"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
)

// RegisteredHost is a page-locked registration of caller-owned host memory.
// Where AllocHost allocates pinned memory, RegisterHost pins a slice the caller
// already owns via cuMemHostRegister, so existing buffers can get pinned-memory
// transfer behavior without a separate allocation.
//
// Lifetime rule: the caller owns the backing memory and must keep the slice
// alive and unchanged (do not resize, reslice the backing array away, or reuse
// it) until Close unregisters it. RegisteredHost retains a reference to the
// slice, so its backing array is not collected while the registration is open,
// but the caller must not free or repurpose it through another alias. Close the
// registration before its owning Context.
type RegisteredHost[T Supported] struct {
	ctx    *Context
	mem    []T
	ptr    *byte
	bytes  uint64
	opMu   sync.RWMutex
	closed bool
}

// RegisterHost page-locks the backing memory of mem so the driver can DMA to and
// from it without staging through a pageable bounce buffer. mem must be
// non-empty. The returned RegisteredHost must be closed before ctx, and returns
// ErrSymbolUnavailable on a driver that does not export cuMemHostRegister.
func RegisterHost[T Supported](ctx *Context, mem []T) (*RegisteredHost[T], error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if len(mem) == 0 {
		return nil, ErrInvalidLength
	}
	ptr := (*byte)(unsafe.Pointer(&mem[0]))
	bytes := uint64(len(mem)) * elemSize[T]()
	err := ctx.doWait(context.Background(), func() error {
		return cudaresult.MemHostRegister(ctx.driver, ptr, bytes, 0)
	})
	runtime.KeepAlive(mem)
	if err != nil {
		return nil, err
	}
	return &RegisteredHost[T]{ctx: ctx, mem: mem, ptr: ptr, bytes: bytes}, nil
}

// Slice returns the registered host slice for use with the copy methods. It is
// nil for a nil receiver.
func (r *RegisteredHost[T]) Slice() []T {
	if r == nil {
		return nil
	}
	return r.mem
}

// Len is the number of elements registered. It is 0 for a nil receiver.
func (r *RegisteredHost[T]) Len() int {
	if r == nil {
		return 0
	}
	return len(r.mem)
}

// Bytes is the registered size in bytes. It is 0 for a nil receiver.
func (r *RegisteredHost[T]) Bytes() uint64 {
	if r == nil {
		return 0
	}
	return r.bytes
}

// Close unregisters the host memory. It is safe to call more than once; a failed
// unregister leaves the registration open so the caller can retry.
func (r *RegisteredHost[T]) Close() error {
	if r == nil {
		return ErrNilBuffer
	}
	r.opMu.Lock()
	defer r.opMu.Unlock()
	if r.closed {
		return nil
	}
	err := r.ctx.doWait(context.Background(), func() error {
		return cudaresult.MemHostUnregister(r.ctx.driver, r.ptr)
	})
	runtime.KeepAlive(r.mem)
	if err != nil {
		return err
	}
	r.closed = true
	return nil
}
