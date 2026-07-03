package cuda

import (
	"testing"
	"unsafe"
)

// TestMemOpRecycleClearsHost guards the memOp recycle path: returning an op to
// the pool must not leave the caller's host pointer reachable.
func TestMemOpRecycleClearsHost(t *testing.T) {
	var x byte
	o := &memOp{host: &x}
	o.recycle()
	if o.host != nil {
		t.Fatal("memOp.recycle left the host pointer set")
	}
}

// TestLaunchOpRecycleClearsParams guards the launchOp recycle path: returning
// an op to the pool must not leave the packed kernel parameter array reachable.
func TestLaunchOpRecycleClearsParams(t *testing.T) {
	var p unsafe.Pointer
	o := &launchOp{params: &p}
	o.recycle()
	if o.params != nil {
		t.Fatal("launchOp.recycle left the params pointer set")
	}
}
