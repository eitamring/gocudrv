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

// CUjitInputType values passed to cuLinkAddData.
const (
	jitInputCubin uint32 = 0
	jitInputPTX   uint32 = 1
)

// Linker is a JIT link session that combines PTX and cubin inputs into a single
// cubin image via cuLink. It is owned by a Context and, like the other handle
// types, locks its operations against a concurrent Close.
type Linker struct {
	ctx *Context
	raw cudasys.CUlinkState
	// Retained and written by the driver for the life of the link state;
	// pinned until a successful Close.
	infoBuf []byte
	errBuf  []byte
	options []int32
	values  []uintptr
	pin     runtime.Pinner
	opMu    sync.RWMutex
	closed  bool
}

// NewLinker starts a JIT link session on the context. opts.LogBufferBytes sizes
// the info and error log buffers (zero for the default, negative or over the cap
// rejected); opts.MaxRegisters caps registers per thread when > 0 and must fit
// in uint32. A driver missing ANY cuLink symbol returns ErrSymbolUnavailable:
// the group is all-or-nothing, or a created state could never be destroyed.
func (c *Context) NewLinker(opts JITOptions) (*Linker, error) {
	if c == nil {
		return nil, ErrNilContext
	}
	d := c.driver
	if d == nil || d.CuLinkCreate == nil || d.CuLinkAddData == nil ||
		d.CuLinkComplete == nil || d.CuLinkDestroy == nil {
		return nil, ErrSymbolUnavailable
	}
	size := opts.LogBufferBytes
	if size < 0 || size > maxJITLogBytes {
		return nil, ErrInvalidLength
	}
	if size == 0 {
		size = defaultJITLogBytes
	}
	if int64(opts.MaxRegisters) > math.MaxUint32 {
		return nil, ErrInvalidValue
	}

	l := &Linker{ctx: c, infoBuf: make([]byte, size), errBuf: make([]byte, size)}
	l.options = []int32{jitInfoLogBuffer, jitInfoLogBufferSizeBytes, jitErrorLogBuffer, jitErrorLogBufferSizeBytes}
	l.values = []uintptr{
		uintptr(unsafe.Pointer(&l.infoBuf[0])),
		uintptr(size),
		uintptr(unsafe.Pointer(&l.errBuf[0])),
		uintptr(size),
	}
	if opts.MaxRegisters > 0 {
		l.options = append(l.options, jitMaxRegisters)
		l.values = append(l.values, uintptr(opts.MaxRegisters))
	}

	l.pin.Pin(&l.infoBuf[0])
	l.pin.Pin(&l.errBuf[0])
	l.pin.Pin(&l.options[0])
	l.pin.Pin(&l.values[0])
	err := c.doWait(context.Background(), func() error {
		state, e := cudaresult.LinkCreate(c.driver, l.options, l.values)
		if e != nil {
			return e
		}
		l.raw = state
		return nil
	})
	if err != nil {
		l.pin.Unpin()
		return nil, err
	}
	return l, nil
}

// AddPTX adds a PTX input labelled name (empty name is unlabelled). The image is
// null-terminated and its length is passed including the terminator, which the
// driver's PTX parser requires.
func (l *Linker) AddPTX(name string, ptx []byte) error {
	return l.addInput(name, ptx, jitInputPTX, true)
}

// AddCubin adds a cubin input labelled name (empty name is unlabelled). The bytes
// are passed as-is with their exact length.
func (l *Linker) AddCubin(name string, cubin []byte) error {
	return l.addInput(name, cubin, jitInputCubin, false)
}

// addInput takes the write lock, not a read lock: the driver writes into the
// log buffers during the call, and Log reads them under the read lock.
func (l *Linker) addInput(name string, data []byte, inputType uint32, terminate bool) error {
	if l == nil {
		return ErrNilLinker
	}
	l.opMu.Lock()
	defer l.opMu.Unlock()
	if l.closed {
		return ErrLinkerClosed
	}
	if len(data) == 0 {
		return ErrEmptyImage
	}
	buf := data
	if terminate {
		buf = nullTerminated(data)
	}
	var namePtr *byte
	var nameBuf []byte
	if name != "" {
		nameBuf = make([]byte, len(name)+1)
		copy(nameBuf, name)
		namePtr = &nameBuf[0]
	}
	err := l.ctx.doWait(context.Background(), func() error {
		return cudaresult.LinkAddData(l.ctx.driver, l.raw, inputType, &buf[0], uint64(len(buf)), namePtr)
	})
	runtime.KeepAlive(buf)
	runtime.KeepAlive(nameBuf)
	return err
}

// Complete finishes the link and returns a fresh copy of the resulting cubin. The
// driver owns the underlying buffer and frees it at Close, so the copy is taken
// inside the driver call before Close can run.
func (l *Linker) Complete() ([]byte, error) {
	if l == nil {
		return nil, ErrNilLinker
	}
	l.opMu.Lock()
	defer l.opMu.Unlock()
	if l.closed {
		return nil, ErrLinkerClosed
	}
	var out []byte
	err := l.ctx.doWait(context.Background(), func() error {
		ptr, size, e := cudaresult.LinkComplete(l.ctx.driver, l.raw)
		if e != nil {
			return e
		}
		out = append([]byte(nil), unsafe.Slice((*byte)(ptr), size)...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Log returns the info and error logs the driver produced during the link, so a
// failed AddPTX or Complete keeps its diagnostics. A nil or never-created Linker
// returns the zero JITLog.
func (l *Linker) Log() JITLog {
	if l == nil {
		return JITLog{}
	}
	l.opMu.RLock()
	defer l.opMu.RUnlock()
	return JITLog{Info: trimNull(l.infoBuf), Error: trimNull(l.errBuf)}
}

// Close destroys the link session and the cubin buffer it owns. Idempotent after
// a successful destroy and safe on a nil receiver.
func (l *Linker) Close() error {
	if l == nil {
		return ErrNilLinker
	}
	l.opMu.Lock()
	defer l.opMu.Unlock()
	if l.closed {
		return nil
	}
	if err := l.ctx.doBarrier(context.Background(), func() error {
		return cudaresult.LinkDestroy(l.ctx.driver, l.raw)
	}); err != nil {
		return err
	}
	l.pin.Unpin()
	l.closed = true
	return nil
}
