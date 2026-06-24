package cuda

import (
	"context"
	"math"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// MemoryPool is a handle to a CUDA stream-ordered memory pool. The default pool
// from Context.DefaultMemPool is owned by the driver, so MemoryPool has no
// Close.
type MemoryPool struct {
	ctx *Context
	raw cudasys.CUmemoryPool
}

// DefaultMemPool returns the default memory pool of the context's device. It
// returns ErrSymbolUnavailable on a driver without stream-ordered memory pools.
func (c *Context) DefaultMemPool() (*MemoryPool, error) {
	if c == nil {
		return nil, ErrNilContext
	}
	var pool cudasys.CUmemoryPool
	err := c.do(context.Background(), func() error {
		p, e := cudaresult.DeviceGetDefaultMemPool(c.driver, c.device.handle)
		if e != nil {
			return e
		}
		pool = p
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &MemoryPool{ctx: c, raw: pool}, nil
}

func (p *MemoryPool) getU64(attr int32) (uint64, error) {
	if p == nil {
		return 0, ErrNilMemPool
	}
	var v uint64
	err := p.ctx.do(context.Background(), func() error {
		got, e := cudaresult.MemPoolGetAttributeU64(p.ctx.driver, p.raw, attr)
		if e != nil {
			return e
		}
		v = got
		return nil
	})
	return v, err
}

// ReleaseThreshold returns the amount of reserved memory in bytes the pool
// holds onto before releasing it back to the OS.
func (p *MemoryPool) ReleaseThreshold() (uint64, error) {
	return p.getU64(cudasys.MemPoolAttrReleaseThreshold)
}

// SetReleaseThreshold sets the release threshold in bytes.
func (p *MemoryPool) SetReleaseThreshold(bytes uint64) error {
	if p == nil {
		return ErrNilMemPool
	}
	return p.ctx.do(context.Background(), func() error {
		return cudaresult.MemPoolSetAttributeU64(p.ctx.driver, p.raw, cudasys.MemPoolAttrReleaseThreshold, bytes)
	})
}

// ReservedMemCurrent returns the total bytes currently reserved by the pool.
func (p *MemoryPool) ReservedMemCurrent() (uint64, error) {
	return p.getU64(cudasys.MemPoolAttrReservedMemCurrent)
}

// UsedMemCurrent returns the total bytes currently allocated from the pool.
func (p *MemoryPool) UsedMemCurrent() (uint64, error) {
	return p.getU64(cudasys.MemPoolAttrUsedMemCurrent)
}

// AllocFromPool allocates n elements of T from pool, ordered on stream, and
// returns a Buffer. As with AllocAsync the memory is ready once the stream
// reaches this point; free it with FreeAsync on a stream, or Close once the
// stream work that uses it has drained. stream must belong to the pool's
// context.
func AllocFromPool[T Supported](pool *MemoryPool, stream *Stream, n int) (*Buffer[T], error) {
	if pool == nil {
		return nil, ErrNilMemPool
	}
	if stream == nil {
		return nil, ErrNilStream
	}
	if n <= 0 {
		return nil, ErrInvalidLength
	}
	es := elemSize[T]()
	if uint64(n) > math.MaxUint64/es {
		return nil, ErrInvalidLength
	}
	bytes := uint64(n) * es

	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return nil, ErrStreamClosed
	}
	if stream.ctx != pool.ctx {
		return nil, ErrContextMismatch
	}
	var ptr cudasys.CUdeviceptr
	err := pool.ctx.doWait(context.Background(), func() error {
		p, e := cudaresult.MemAllocFromPoolAsync(pool.ctx.driver, bytes, pool.raw, stream.raw)
		if e != nil {
			return e
		}
		ptr = p
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Buffer[T]{ctx: pool.ctx, ptr: ptr, length: n, bytes: bytes}, nil
}
