package cuda

import (
	"context"
	"runtime"
	"strings"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// Global is a handle to a __device__ or __constant__ global variable in a
// loaded Module. Its lifetime is bound to the Module: once the Module is
// closed the handle is invalid.
type Global struct {
	module *Module
	ptr    cudasys.CUdeviceptr
	bytes  uint64
	name   string
}

// Global looks up a __device__ or __constant__ global by name and returns a
// handle for reading and writing it. The name is converted to a
// null-terminated byte sequence before being passed to cuModuleGetGlobal.
func (m *Module) Global(name string) (*Global, error) {
	if m == nil {
		return nil, ErrNilModule
	}
	if name == "" {
		return nil, ErrEmptyGlobalName
	}
	if strings.IndexByte(name, 0) >= 0 {
		return nil, ErrInvalidGlobalName
	}
	m.opMu.RLock()
	defer m.opMu.RUnlock()
	if m.closed {
		return nil, ErrModuleClosed
	}
	nameBuf := make([]byte, len(name)+1)
	copy(nameBuf, name)

	var ptr cudasys.CUdeviceptr
	var bytes uint64
	err := m.ctx.doWait(context.Background(), func() error {
		p, b, e := cudaresult.ModuleGetGlobal(m.ctx.driver, m.raw, (*byte)(unsafe.Pointer(&nameBuf[0])))
		if e != nil {
			return e
		}
		ptr, bytes = p, b
		return nil
	})
	runtime.KeepAlive(nameBuf)
	if err != nil {
		return nil, err
	}
	return &Global{module: m, ptr: ptr, bytes: bytes, name: name}, nil
}

// Bytes returns the size of the global in bytes. Returns 0 for a nil receiver.
func (g *Global) Bytes() uint64 {
	if g == nil {
		return 0
	}
	return g.bytes
}

// Name returns the symbol name this Global was looked up with. Returns the
// empty string for a nil receiver.
func (g *Global) Name() string {
	if g == nil {
		return ""
	}
	return g.name
}

// WriteGlobal copies vals from the host into the device global g. The byte
// size of vals must be greater than zero and must not exceed g.Bytes(). Blocks
// until the copy completes. Cancellation semantics match Buffer.CopyFrom.
func WriteGlobal[T Supported](ctx context.Context, g *Global, vals []T) error {
	if g == nil {
		return ErrNilGlobal
	}
	if g.module == nil {
		return ErrNilModule
	}
	if len(vals) == 0 {
		return ErrLengthMismatch
	}
	var zero T
	bytes := uint64(len(vals)) * uint64(unsafe.Sizeof(zero))
	g.module.opMu.RLock()
	defer g.module.opMu.RUnlock()
	if g.module.closed {
		return ErrModuleClosed
	}
	if bytes > g.bytes {
		return ErrLengthMismatch
	}
	srcPtr := (*byte)(unsafe.Pointer(&vals[0]))
	err := g.module.ctx.memcpyHtoD(ctx, g.ptr, srcPtr, bytes)
	runtime.KeepAlive(vals)
	return err
}

// ReadGlobal copies the device global g into dst. The byte size of dst must be
// greater than zero and must not exceed g.Bytes(). Blocks until the copy
// completes. Cancellation semantics match Buffer.CopyTo.
func ReadGlobal[T Supported](ctx context.Context, dst []T, g *Global) error {
	if g == nil {
		return ErrNilGlobal
	}
	if g.module == nil {
		return ErrNilModule
	}
	if len(dst) == 0 {
		return ErrLengthMismatch
	}
	var zero T
	bytes := uint64(len(dst)) * uint64(unsafe.Sizeof(zero))
	g.module.opMu.RLock()
	defer g.module.opMu.RUnlock()
	if g.module.closed {
		return ErrModuleClosed
	}
	if bytes > g.bytes {
		return ErrLengthMismatch
	}
	dstPtr := (*byte)(unsafe.Pointer(&dst[0]))
	err := g.module.ctx.memcpyDtoH(ctx, dstPtr, g.ptr, bytes)
	runtime.KeepAlive(dst)
	return err
}
