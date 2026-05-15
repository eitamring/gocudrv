package argpack

import (
	"runtime"
	"unsafe"
)

const inlineArgs = 16

// Builder owns temporary kernel argument storage until the launch call returns.
// CUDA receives an array of pointers to these values, not the values directly.
type Builder struct {
	inlineValues   [inlineArgs]uint64
	inlinePointers [inlineArgs]unsafe.Pointer
	count          int
	spillValues    []any
	spillPointers  []unsafe.Pointer
}

// Add stores v and appends a pointer to its stable storage. Values up to eight
// bytes use inline storage for the common path; larger values or launches with
// many arguments spill to heap-backed storage.
func Add[T any](b *Builder, v T) {
	size := unsafe.Sizeof(v)
	if b.count < inlineArgs && size <= unsafe.Sizeof(uint64(0)) {
		slot := &b.inlineValues[b.count]
		*slot = 0
		copy(
			unsafe.Slice((*byte)(unsafe.Pointer(slot)), size),
			unsafe.Slice((*byte)(unsafe.Pointer(&v)), size),
		)
		b.inlinePointers[b.count] = unsafe.Pointer(slot)
		b.count++
		return
	}

	p := new(T)
	*p = v
	b.spillValues = append(b.spillValues, p)
	if b.spillPointers == nil {
		b.spillPointers = append(b.spillPointers, b.inlinePointers[:b.count]...)
	}
	b.spillPointers = append(b.spillPointers, unsafe.Pointer(p))
	b.count++
}

// Params returns the kernel parameter pointer array expected by cuLaunchKernel.
// It returns nil when no parameters were added.
func (b *Builder) Params() *unsafe.Pointer {
	if b.count == 0 {
		return nil
	}
	if b.spillPointers != nil {
		return &b.spillPointers[0]
	}
	return &b.inlinePointers[0]
}

// Len returns the number of packed kernel arguments.
func (b *Builder) Len() int {
	return b.count
}

// KeepAlive retains the temporary storage until after the foreign call returns.
func (b *Builder) KeepAlive() {
	runtime.KeepAlive(b)
	runtime.KeepAlive(b.spillValues)
	runtime.KeepAlive(b.spillPointers)
}
