// Package hostcb bridges CUDA host-function callbacks into Go. The driver
// needs a C function pointer it can call from its own internal thread; foreign
// callback pointers can never be released, so the package mints exactly one
// trampoline and dispatches through a keyed registry of Go closures.
package hostcb

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"github.com/ebitengine/purego"
)

var (
	registry sync.Map
	nextKey  atomic.Uintptr
	tramp    uintptr
	trampMu  sync.Mutex
)

// Register stores fn and returns the key to pass as the callback's user data.
// The entry is removed when the callback fires; Unregister covers the enqueue-
// failed path.
func Register(fn func()) uintptr {
	k := nextKey.Add(1)
	registry.Store(k, fn)
	return k
}

// Unregister drops a key whose callback will never fire (failed enqueue).
func Unregister(key uintptr) {
	registry.Delete(key)
}

// Invoke runs and removes the closure for key. It is the trampoline body,
// exported so tests can exercise dispatch without a foreign caller. Panics are
// reported to stderr and swallowed: the caller is a non-Go driver thread,
// where an escaping panic would kill the process.
func Invoke(key uintptr) {
	v, ok := registry.LoadAndDelete(key)
	if !ok {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "cuda: panic in host function: %v\n", r)
		}
	}()
	v.(func())()
}

// Callback returns the process-wide C-callable trampoline pointer, creating it
// on first use.
func Callback() uintptr {
	trampMu.Lock()
	defer trampMu.Unlock()
	if tramp == 0 {
		tramp = purego.NewCallback(func(userData uintptr) uintptr {
			Invoke(userData)
			return 0
		})
	}
	return tramp
}

// Pending reports whether key is still registered (test helper).
func Pending(key uintptr) bool {
	_, ok := registry.Load(key)
	return ok
}
