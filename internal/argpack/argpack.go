package argpack

import (
	"errors"
	"reflect"
	"runtime"
	"unsafe"
)

const inlineArgs = 16

// spillAlign aligns every arena argument; combined with minArenaCap it keeps
// each spilled argument pointer at least 8-byte aligned.
const spillAlign = 8

// minArenaCap keeps the arena out of the tiny allocator so its base address
// is 8-byte aligned.
const minArenaCap = 64

// MaxRetainedSpillBytes caps the arena capacity Reset keeps, so pooled
// builders do not pin unusually large arguments forever. It is twice the
// largest single raw argument the cuda layer accepts.
const MaxRetainedSpillBytes = 8192

var (
	ErrIndexOutOfRange = errors.New("cuda: packed argument index out of range")
	ErrTypeMismatch    = errors.New("cuda: packed argument type mismatch")
	ErrSizeMismatch    = errors.New("cuda: packed argument size mismatch")
)

// argMeta records where one packed argument lives and how updates are checked.
// A negative offset means the argument's own inline slot; typ is nil for raw
// byte arguments, which are checked by size only.
type argMeta struct {
	offset int32
	size   uint16
	typ    reflect.Type
}

// Builder owns kernel argument storage until the launch call returns and
// across in-place updates. CUDA receives an array of pointers to these
// values, not the values directly. Reset keeps grown storage for reuse.
type Builder struct {
	inlineValues   [inlineArgs]uint64
	inlinePointers [inlineArgs]unsafe.Pointer
	inlineMeta     [inlineArgs]argMeta
	overflowMeta   []argMeta
	count          int
	spillBuf       []byte
	spillPointers  []unsafe.Pointer
	hasSpill       bool
	dirty          bool
}

// Add stores v and records a pointer to its stable storage: inline slots for
// the first sixteen args of eight bytes or less, the arena otherwise. T must
// not contain Go pointers; argument bytes are invisible to the GC.
func Add[T any](b *Builder, v T) {
	size := unsafe.Sizeof(v)
	b.add(unsafe.Slice((*byte)(unsafe.Pointer(&v)), size), reflect.TypeFor[T]())
	runtime.KeepAlive(v)
}

// AddBytes stores a copy of data and records a pointer to it, for kernel
// argument values whose type is not known at compile time. data must be
// non-empty; the copy is owned by the Builder.
func AddBytes(b *Builder, data []byte) {
	b.add(data, nil)
}

func (b *Builder) add(data []byte, typ reflect.Type) {
	size := len(data)
	if b.count < inlineArgs && size <= 8 {
		slot := &b.inlineValues[b.count]
		*slot = 0
		copy(unsafe.Slice((*byte)(unsafe.Pointer(slot)), size), data)
		b.inlinePointers[b.count] = unsafe.Pointer(slot)
		b.setMeta(b.count, argMeta{offset: -1, size: uint16(size), typ: typ})
		b.count++
		b.dirty = b.hasSpill
		return
	}

	off := b.arenaAlloc(size)
	copy(b.spillBuf[off:off+size], data)
	b.setMeta(b.count, argMeta{offset: int32(off), size: uint16(size), typ: typ})
	b.count++
	b.hasSpill = true
	b.dirty = true
}

func (b *Builder) arenaAlloc(size int) int {
	off := (len(b.spillBuf) + spillAlign - 1) &^ (spillAlign - 1)
	need := off + size
	if cap(b.spillBuf) < need {
		grownCap := 2 * cap(b.spillBuf)
		if grownCap < need {
			grownCap = need
		}
		if grownCap < minArenaCap {
			grownCap = minArenaCap
		}
		grown := make([]byte, need, grownCap)
		copy(grown, b.spillBuf)
		b.spillBuf = grown
		return off
	}
	b.spillBuf = b.spillBuf[:need]
	return off
}

func (b *Builder) setMeta(i int, m argMeta) {
	if i < inlineArgs {
		b.inlineMeta[i] = m
		return
	}
	b.overflowMeta = append(b.overflowMeta, m)
}

func (b *Builder) meta(i int) *argMeta {
	if i < inlineArgs {
		return &b.inlineMeta[i]
	}
	return &b.overflowMeta[i-inlineArgs]
}

// Set updates argument i in place with a value of the exact type it was
// packed with. Raw byte arguments have no recorded type and reject Set.
func Set[T any](b *Builder, i int, v T) error {
	if i < 0 || i >= b.count {
		return ErrIndexOutOfRange
	}
	m := b.meta(i)
	if m.typ == nil || m.typ != reflect.TypeFor[T]() {
		return ErrTypeMismatch
	}
	b.write(m, i, unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v)))
	runtime.KeepAlive(v)
	return nil
}

// SetBytes updates argument i in place from data, checking size but not type,
// so it can also reinterpret a typed slot of the same width.
func (b *Builder) SetBytes(i int, data []byte) error {
	if i < 0 || i >= b.count {
		return ErrIndexOutOfRange
	}
	m := b.meta(i)
	if len(data) != int(m.size) {
		return ErrSizeMismatch
	}
	b.write(m, i, data)
	return nil
}

// write requires len(data) == m.size; slots are never partially overwritten.
func (b *Builder) write(m *argMeta, i int, data []byte) {
	if m.offset < 0 {
		slot := &b.inlineValues[i]
		copy(unsafe.Slice((*byte)(unsafe.Pointer(slot)), len(data)), data)
		return
	}
	copy(b.spillBuf[m.offset:int(m.offset)+len(data)], data)
}

// Params returns the kernel parameter pointer array expected by cuLaunchKernel.
// It returns nil when no parameters were added.
func (b *Builder) Params() *unsafe.Pointer {
	if b.count == 0 {
		return nil
	}
	if !b.hasSpill {
		return &b.inlinePointers[0]
	}
	if b.dirty {
		b.spillPointers = b.spillPointers[:0]
		for i := 0; i < b.count; i++ {
			m := b.meta(i)
			if m.offset < 0 {
				b.spillPointers = append(b.spillPointers, b.inlinePointers[i])
				continue
			}
			b.spillPointers = append(b.spillPointers, unsafe.Pointer(&b.spillBuf[m.offset]))
		}
		b.dirty = false
	}
	return &b.spillPointers[0]
}

// Len returns the number of packed kernel arguments.
func (b *Builder) Len() int {
	return b.count
}

// Reset clears the arguments while keeping grown storage for reuse. An arena
// past MaxRetainedSpillBytes is dropped with its pointer and metadata slices;
// each overflow argument uses arena bytes, so the cap bounds all three.
func (b *Builder) Reset() {
	b.count = 0
	b.hasSpill = false
	b.dirty = false
	if cap(b.spillBuf) > MaxRetainedSpillBytes {
		b.spillBuf = nil
		b.spillPointers = nil
		b.overflowMeta = nil
		return
	}
	b.spillBuf = b.spillBuf[:0]
	b.spillPointers = b.spillPointers[:0]
	b.overflowMeta = b.overflowMeta[:0]
}

// KeepAlive retains the argument storage until after the foreign call returns.
func (b *Builder) KeepAlive() {
	runtime.KeepAlive(b)
	runtime.KeepAlive(b.spillBuf)
	runtime.KeepAlive(b.spillPointers)
}
