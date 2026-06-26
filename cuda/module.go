package cuda

import (
	"bytes"
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
	buf := nullTerminated(image)

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

// nullTerminated returns image unchanged if it already ends in a null byte, or a
// fresh null-terminated copy otherwise, so the original slice is never mutated.
// PTX images must be null-terminated for cuModuleLoadData(Ex).
func nullTerminated(image []byte) []byte {
	if len(image) > 0 && image[len(image)-1] == 0 {
		return image
	}
	buf := make([]byte, len(image)+1)
	copy(buf, image)
	return buf
}

// trimNull returns the bytes of b up to the first null, as a string.
func trimNull(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		return string(b[:i])
	}
	return string(b)
}

// JITOptions tunes a JIT compile in LoadModuleEx. The zero value requests info
// and error logs at a default buffer size with no other tuning.
type JITOptions struct {
	// LogBufferBytes is the size of each of the info and error log buffers. A
	// value <= 0 uses a default size.
	LogBufferBytes int
	// MaxRegisters caps registers per thread (CU_JIT_MAX_REGISTERS). 0 leaves it
	// at the driver default.
	MaxRegisters int
}

// JITLog holds the info and error logs the driver produced while loading a
// module. Error is populated even when the load fails.
type JITLog struct {
	Info  string
	Error string
}

const defaultJITLogBytes = 8192

// CUjit_option values used by LoadModuleEx.
const (
	jitMaxRegisters            int32 = 0
	jitInfoLogBuffer           int32 = 3
	jitInfoLogBufferSizeBytes  int32 = 4
	jitErrorLogBuffer          int32 = 5
	jitErrorLogBufferSizeBytes int32 = 6
)

// LoadModuleEx loads a module like LoadModule but with JIT options, returning
// the driver's info and error logs. The error log is filled even when the load
// fails, so a PTX compile error surfaces useful diagnostics. Use LoadModule for
// the simple case.
func (c *Context) LoadModuleEx(image []byte, opts JITOptions) (*Module, JITLog, error) {
	if c == nil {
		return nil, JITLog{}, ErrNilContext
	}
	if len(image) == 0 {
		return nil, JITLog{}, ErrEmptyImage
	}
	buf := nullTerminated(image)

	size := opts.LogBufferBytes
	if size <= 0 {
		size = defaultJITLogBytes
	}
	infoBuf := make([]byte, size)
	errBuf := make([]byte, size)
	options := []int32{jitInfoLogBuffer, jitInfoLogBufferSizeBytes, jitErrorLogBuffer, jitErrorLogBufferSizeBytes}
	values := []uintptr{
		uintptr(unsafe.Pointer(&infoBuf[0])),
		uintptr(size),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(size),
	}
	if opts.MaxRegisters > 0 {
		options = append(options, jitMaxRegisters)
		values = append(values, uintptr(opts.MaxRegisters))
	}

	var raw cudasys.CUmodule
	err := c.doWait(context.Background(), func() error {
		m, e := cudaresult.ModuleLoadDataEx(c.driver, (*byte)(unsafe.Pointer(&buf[0])), options, values)
		if e != nil {
			return e
		}
		raw = m
		return nil
	})
	runtime.KeepAlive(buf)
	runtime.KeepAlive(infoBuf)
	runtime.KeepAlive(errBuf)

	log := JITLog{Info: trimNull(infoBuf), Error: trimNull(errBuf)}
	if err != nil {
		return nil, log, err
	}
	return &Module{ctx: c, raw: raw}, log, nil
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
