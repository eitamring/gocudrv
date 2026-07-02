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

func TestSpillThenInlineOrdering(t *testing.T) {
	var b Builder
	big := [16]byte{1, 2, 3}
	Add(&b, big)
	Add(&b, int32(7))
	if b.Len() != 2 {
		t.Fatalf("Len = %d, want 2", b.Len())
	}
	params := unsafe.Slice(b.Params(), b.Len())
	if got := unsafe.Slice((*byte)(params[0]), 16); got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("arg0 = %v, want the 16-byte value", got[:3])
	}
	if got := *(*int32)(params[1]); got != 7 {
		t.Errorf("arg1 = %d, want 7 (small arg lost after spill fork)", got)
	}
	b.KeepAlive()
}

type packStep struct {
	small uint64
	big   [16]byte
	isBig bool
	bytes []byte
}

func small(v uint64) packStep     { return packStep{small: v} }
func big(first byte) packStep     { return packStep{big: [16]byte{first, 0xBB}, isBig: true} }
func rawBytes(bs []byte) packStep { return packStep{bytes: bs} }

// TestOrderingCombinations packs mixed small, large, and byte arguments in
// orders that cross the inline-to-spill fork in both directions and verifies
// every parameter pointer dereferences to the right value in the right slot.
func TestOrderingCombinations(t *testing.T) {
	overflow := make([]packStep, 0, inlineArgs+3)
	for i := 0; i < inlineArgs; i++ {
		overflow = append(overflow, small(uint64(100+i)))
	}
	overflow = append(overflow, big(0xA1), small(42), rawBytes([]byte{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2}))

	cases := []struct {
		name  string
		steps []packStep
	}{
		{"big then small", []packStep{big(0xA1), small(7)}},
		{"big then small bytes", []packStep{big(0xE5), rawBytes([]byte{1, 2, 3})}},
		{"small big small", []packStep{small(1), big(0xB2), small(3)}},
		{"big small big small", []packStep{big(0xC3), small(2), big(0xD4), small(4)}},
		{"bytes-spill then smalls", []packStep{rawBytes([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}), small(5), small(6)}},
		{"small then bytes-spill then small", []packStep{small(8), rawBytes([]byte{4, 4, 4, 4, 4, 4, 4, 4, 4, 4}), small(9)}},
		{"inline overflow then big then small then bytes", overflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var b Builder
			for _, s := range tc.steps {
				switch {
				case s.isBig:
					Add(&b, s.big)
				case s.bytes != nil:
					AddBytes(&b, s.bytes)
				default:
					Add(&b, s.small)
				}
			}
			if b.Len() != len(tc.steps) {
				t.Fatalf("Len = %d, want %d", b.Len(), len(tc.steps))
			}
			params := unsafe.Slice(b.Params(), b.Len())
			for i, s := range tc.steps {
				switch {
				case s.isBig:
					got := unsafe.Slice((*byte)(params[i]), 16)
					if got[0] != s.big[0] || got[1] != 0xBB {
						t.Errorf("arg%d = %v, want big value %#x", i, got[:2], s.big[0])
					}
				case s.bytes != nil:
					got := unsafe.Slice((*byte)(params[i]), len(s.bytes))
					for j := range s.bytes {
						if got[j] != s.bytes[j] {
							t.Fatalf("arg%d byte[%d] = %d, want %d", i, j, got[j], s.bytes[j])
						}
					}
				default:
					if got := *(*uint64)(params[i]); got != s.small {
						t.Errorf("arg%d = %d, want %d", i, got, s.small)
					}
				}
			}
			b.KeepAlive()
		})
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
