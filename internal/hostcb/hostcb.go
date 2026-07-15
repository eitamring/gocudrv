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
	registry  [registryShardCount]callbackShard
	nextKey   atomic.Uintptr
	tramp     uintptr
	trampOnce sync.Once
)

const registryShardCount = 32

type callbackShard struct {
	mu      sync.Mutex
	pending map[uintptr]func()
}

func shardFor(key uintptr) *callbackShard {
	return &registry[key%registryShardCount]
}

// Register stores fn and returns the key to pass as the callback's user data.
// The entry is removed when the callback fires; Unregister covers the enqueue-
// failed path.
func Register(fn func()) uintptr {
	k := nextKey.Add(1)
	shard := shardFor(k)
	shard.mu.Lock()
	if shard.pending == nil {
		shard.pending = make(map[uintptr]func())
	}
	shard.pending[k] = fn
	shard.mu.Unlock()
	return k
}

// Unregister drops a key whose callback will never fire (failed enqueue).
func Unregister(key uintptr) {
	shard := shardFor(key)
	shard.mu.Lock()
	delete(shard.pending, key)
	shard.mu.Unlock()
}

// Invoke runs and removes the closure for key. It is the trampoline body,
// exported so tests can exercise dispatch without a foreign caller. Panics are
// reported to stderr and swallowed: the caller is a non-Go driver thread,
// where an escaping panic would kill the process.
func Invoke(key uintptr) {
	shard := shardFor(key)
	shard.mu.Lock()
	fn, ok := shard.pending[key]
	if ok {
		delete(shard.pending, key)
	}
	shard.mu.Unlock()
	if !ok {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "cuda: panic in host function: %v\n", r)
		}
	}()
	fn()
}

func callback(userData uintptr) uintptr {
	Invoke(userData)
	return 0
}

func initCallback() {
	tramp = purego.NewCallback(callback)
}

// Callback returns the process-wide C-callable trampoline pointer, creating it
// on first use.
func Callback() uintptr {
	trampOnce.Do(initCallback)
	return tramp
}

// Pending reports whether key is still registered (test helper).
func Pending(key uintptr) bool {
	shard := shardFor(key)
	shard.mu.Lock()
	_, ok := shard.pending[key]
	shard.mu.Unlock()
	return ok
}
