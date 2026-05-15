package argpack

import (
	"testing"
	"unsafe"
)

func TestBuilderAddAndParams(t *testing.T) {
	var b Builder
	if got := b.Params(); got != nil {
		t.Errorf("empty Params = %p, want nil", got)
	}

	Add(&b, int32(7))
	Add(&b, uint64(0xCAFE))

	if got := b.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2", got)
	}
	params := unsafe.Slice(b.Params(), b.Len())
	if got := *(*int32)(params[0]); got != 7 {
		t.Errorf("arg0 = %d, want 7", got)
	}
	if got := *(*uint64)(params[1]); got != 0xCAFE {
		t.Errorf("arg1 = %#x, want 0xCAFE", got)
	}
	b.KeepAlive()
}
