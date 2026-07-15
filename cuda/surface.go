package cuda

import (
	"context"
	"sync"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/argpack"
)

// Surface is a writable view over an Array2D: a kernel reads and writes the
// array through it with surf2Dread and surf2Dwrite. Pass it to a kernel with
// ArgSurface; close it before the array it accesses.
type Surface struct {
	ctx    *Context
	handle cudasys.CUsurfObject
	opMu   sync.RWMutex
	closed bool
}

// NewSurface creates a surface object over arr, which must have been allocated
// with WithSurfaceStore (else ErrNoSurfaceStore). The array must stay open for
// the surface's lifetime. Returns ErrSymbolUnavailable on a driver without the
// surface symbols.
func NewSurface[T Supported](arr *Array2D[T]) (*Surface, error) {
	if arr == nil {
		return nil, ErrNilArray
	}
	arr.opMu.RLock()
	defer arr.opMu.RUnlock()
	if arr.closed {
		return nil, ErrArrayClosed
	}
	if !arr.surface {
		return nil, ErrNoSurfaceStore
	}

	resDesc := cudasys.CUDA_RESOURCE_DESC{
		ResType: cudasys.ResourceTypeArray,
		Handle:  arr.handle,
	}
	var handle cudasys.CUsurfObject
	err := arr.ctx.do(context.Background(), func() error {
		h, e := cudaresult.SurfObjectCreate(arr.ctx.driver, &resDesc)
		if e != nil {
			return e
		}
		handle = h
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Surface{ctx: arr.ctx, handle: handle}, nil
}

// Raw is the underlying CUsurfObject handle (the value a kernel receives as a
// cudaSurfaceObject_t), a snapshot valid only while the surface is open. It is
// 0 for a nil receiver.
func (s *Surface) Raw() cudasys.CUsurfObject {
	if s == nil {
		return 0
	}
	return s.handle
}

// Close destroys the surface object with cuSurfObjectDestroy. Idempotent; a
// failed destroy leaves the surface open to retry.
func (s *Surface) Close() error {
	if s == nil {
		return ErrNilSurface
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	if s.closed {
		return nil
	}
	if err := s.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.SurfObjectDestroy(s.ctx.driver, s.handle)
	}); err != nil {
		return err
	}
	s.closed = true
	return nil
}

type surfaceKernelArg struct {
	s *Surface
}

// ArgSurface passes a surface object to a kernel (a cudaSurfaceObject_t
// parameter). Like Arg it holds the surface's read lock across submission; the
// underlying array must stay open until the launch has been synchronized.
func ArgSurface(s *Surface) KernelArg {
	return surfaceKernelArg{s: s}
}

func (a surfaceKernelArg) appendKernelArg(b *kernelArgBuilder) error {
	if a.s == nil {
		return ErrNilSurface
	}
	held := b.holdsLock(&a.s.opMu)
	if !held {
		a.s.opMu.RLock()
	}
	if a.s.closed {
		if !held {
			a.s.opMu.RUnlock()
		}
		return ErrSurfaceClosed
	}
	if b.ctx != nil && a.s.ctx != b.ctx {
		if !held {
			a.s.opMu.RUnlock()
		}
		return ErrContextMismatch
	}
	handle := a.s.handle
	if b.snapshot {
		if !held {
			a.s.opMu.RUnlock()
		}
	} else if !held {
		b.addLock(&a.s.opMu)
	}
	argpack.Add(b.packed, handle)
	return nil
}
