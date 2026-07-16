//go:build linux

package executor

import (
	"os"
	"syscall"
	"testing"
	"time"
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

func TestRetiredThreadStaysQuarantined(t *testing.T) {
	e := New()
	var tid int
	if err := e.Do(func() error {
		tid = syscall.Gettid()
		return nil
	}); err != nil {
		t.Fatalf("Do: %v", err)
	}
	e.Retire()
	closed := make(chan error, 1)
	go func() { closed <- e.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close blocked on quarantined thread")
	}

	deadline := time.Now().Add(100 * time.Millisecond)
	for time.Now().Before(deadline) {
		if err := syscall.Tgkill(os.Getpid(), tid, 0); err != nil {
			t.Fatalf("quarantined thread %d exited: %v", tid, err)
		}
		time.Sleep(time.Millisecond)
	}
}
