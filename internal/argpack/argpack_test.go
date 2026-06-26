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

func TestAddBytesInline(t *testing.T) {
	var b Builder
	want := uint32(0x11223344)
	AddBytes(&b, unsafe.Slice((*byte)(unsafe.Pointer(&want)), 4))
	if b.Len() != 1 {
		t.Fatalf("Len = %d, want 1", b.Len())
	}
	params := unsafe.Slice(b.Params(), b.Len())
	got := *(*uint32)(params[0])
	if got != want {
		t.Errorf("packed = %#x, want %#x", got, want)
	}
}

func TestAddBytesSpill(t *testing.T) {
	var b Builder
	// A 16-byte value cannot use the inline 8-byte slot, so it spills.
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	Add(&b, int32(1)) // one inline arg first
	AddBytes(&b, data)
	if b.Len() != 2 {
		t.Fatalf("Len = %d, want 2", b.Len())
	}
	params := unsafe.Slice(b.Params(), b.Len())
	got := unsafe.Slice((*byte)(params[1]), len(data))
	for i := range data {
		if got[i] != data[i] {
			t.Fatalf("byte[%d] = %d, want %d", i, got[i], data[i])
		}
	}
	b.KeepAlive()
}
