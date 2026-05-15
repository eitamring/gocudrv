package cuda

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// Module is a handle to a loaded CUDA module (PTX or cubin image) owned by
// a Context.
//
// Lifetime rule: a Module must be closed before its owning Context is
// closed. After the Context is closed, Module.Close cannot reach the
// executor and returns ErrContextClosed; the underlying module is reclaimed
// when the primary context retain count drops to zero, but the wrapper
// cannot guarantee that ordering. Pair every LoadModule with a deferred
// Close and close every module before its Context.
type Module struct {
	ctx    *Context
	raw    cudasys.CUmodule
	opMu   sync.RWMutex
	closed bool
}

// Function is a handle to a kernel within a loaded Module. Its lifetime is
// bound to the Module: once the Module is closed the handle is invalid.
type Function struct {
	module *Module
	raw    cudasys.CUfunction
	name   string
}

// LoadModule loads a PTX or cubin image into the context. The image is
// passed to cuModuleLoadData. PTX images must be null-terminated; if image
// is not already null-terminated, a null byte is appended to a fresh copy
// before submission so the original slice is not mutated.
func (c *Context) LoadModule(image []byte) (*Module, error) {
	if c == nil {
		return nil, ErrNilContext
	}
	if len(image) == 0 {
		return nil, ErrEmptyImage
	}
	buf := image
	if buf[len(buf)-1] != 0 {
		buf = make([]byte, len(image)+1)
		copy(buf, image)
	}

	var raw cudasys.CUmodule
	err := c.doWait(context.Background(), func() error {
		m, e := cudaresult.ModuleLoadData(c.driver, (*byte)(unsafe.Pointer(&buf[0])))
		if e != nil {
			return e
		}
		raw = m
		return nil
	})
	runtime.KeepAlive(buf)
	if err != nil {
		return nil, err
	}
	return &Module{ctx: c, raw: raw}, nil
}

// LoadModuleFromFile reads path and forwards the bytes to LoadModule. An
// empty path is rejected with ErrEmptyImage; read errors are wrapped with
// the path for context.
func (c *Context) LoadModuleFromFile(path string) (*Module, error) {
	if c == nil {
		return nil, ErrNilContext
	}
	if path == "" {
		return nil, ErrEmptyImage
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cuda: read module file %q: %w", path, err)
	}
	return c.LoadModule(data)
}

// Function looks up a kernel by name in the module. The name is converted
// to a null-terminated byte sequence before being passed to
// cuModuleGetFunction.
func (m *Module) Function(name string) (*Function, error) {
	if m == nil {
		return nil, ErrNilModule
	}
	if name == "" {
		return nil, ErrEmptyFunctionName
	}
	if strings.IndexByte(name, 0) >= 0 {
		return nil, ErrInvalidFunctionName
	}
	m.opMu.RLock()
	defer m.opMu.RUnlock()
	if m.closed {
		return nil, ErrModuleClosed
	}
	nameBuf := make([]byte, len(name)+1)
	copy(nameBuf, name)

	var raw cudasys.CUfunction
	err := m.ctx.doWait(context.Background(), func() error {
		f, e := cudaresult.ModuleGetFunction(m.ctx.driver, m.raw, (*byte)(unsafe.Pointer(&nameBuf[0])))
		if e != nil {
			return e
		}
		raw = f
		return nil
	})
	runtime.KeepAlive(nameBuf)
	if err != nil {
		return nil, err
	}
	return &Function{module: m, raw: raw, name: name}, nil
}

// Close unloads the module. Idempotent after a successful unload; failures
// leave the module open so callers can retry. Returns ErrContextClosed if
// the owning Context was closed first.
func (m *Module) Close() error {
	if m == nil {
		return ErrNilModule
	}
	m.opMu.Lock()
	defer m.opMu.Unlock()
	if m.closed {
		return nil
	}
	if err := m.ctx.doWait(context.Background(), func() error {
		return cudaresult.ModuleUnload(m.ctx.driver, m.raw)
	}); err != nil {
		return err
	}
	m.closed = true
	return nil
}

// Name returns the kernel name this Function was looked up with. Returns
// the empty string for a nil receiver.
func (f *Function) Name() string {
	if f == nil {
		return ""
	}
	return f.name
}
