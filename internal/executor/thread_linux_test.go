//go:build linux

package executor

import (
	"syscall"
	"testing"
)

// TestSameOSThread is a linux-only confidence check that all Do calls run
// on the same OS thread, using gettid(2). Other platforms rely on the
// runtime.LockOSThread contract.
func TestSameOSThread(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	seen := make(map[int]struct{})
	for i := 0; i < 10; i++ {
		var tid int
		if err := e.Do(func() error {
			tid = syscall.Gettid()
			return nil
		}); err != nil {
			t.Fatalf("Do: %v", err)
		}
		seen[tid] = struct{}{}
	}
	if len(seen) != 1 {
		t.Errorf("saw %d distinct tids %v, want 1", len(seen), seen)
	}
}
