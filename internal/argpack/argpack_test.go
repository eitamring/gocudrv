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

func TestBuilderSpillsAfterInlineCapacity(t *testing.T) {
	var b Builder
	for i := 0; i < inlineArgs+1; i++ {
		Add(&b, uint32(i))
	}
	if got := b.Len(); got != inlineArgs+1 {
		t.Fatalf("Len = %d, want %d", got, inlineArgs+1)
	}
	params := unsafe.Slice(b.Params(), b.Len())
	for i := range params {
		if got := *(*uint32)(params[i]); got != uint32(i) {
			t.Errorf("arg%d = %d, want %d", i, got, i)
		}
	}
}
