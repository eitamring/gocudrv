package argpack

import (
	"runtime"
	"unsafe"
)

// Builder owns temporary kernel argument storage until the launch call returns.
// CUDA receives an array of pointers to these values, not the values directly.
type Builder struct {
	keepAlive []any
	pointers  []unsafe.Pointer
}

// Add stores v and appends a pointer to its stable heap allocation.
func Add[T any](b *Builder, v T) {
	p := new(T)
	*p = v
	b.keepAlive = append(b.keepAlive, p)
	b.pointers = append(b.pointers, unsafe.Pointer(p))
}

// Params returns the kernel parameter pointer array expected by cuLaunchKernel.
// It returns nil when no parameters were added.
func (b *Builder) Params() *unsafe.Pointer {
	if len(b.pointers) == 0 {
		return nil
	}
	return &b.pointers[0]
}

// Len returns the number of packed kernel arguments.
func (b *Builder) Len() int {
	return len(b.pointers)
}

// KeepAlive retains the temporary storage until after the foreign call returns.
func (b *Builder) KeepAlive() {
	runtime.KeepAlive(b.keepAlive)
	runtime.KeepAlive(b.pointers)
}
