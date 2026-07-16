package argpack

import (
	"errors"
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

func TestSetUpdatesValues(t *testing.T) {
	var b Builder
	Add(&b, int32(7))
	var big [16]byte
	big[0] = 0xA1
	Add(&b, big)
	for i := 0; i < inlineArgs; i++ {
		Add(&b, uint64(100+i))
	}
	params := unsafe.Slice(b.Params(), b.Len())
	before := make([]unsafe.Pointer, len(params))
	copy(before, params)

	if err := Set(&b, 0, int32(42)); err != nil {
		t.Fatalf("Set inline: %v", err)
	}
	var newBig [16]byte
	newBig[0] = 0xB2
	if err := Set(&b, 1, newBig); err != nil {
		t.Fatalf("Set arena: %v", err)
	}
	if err := Set(&b, 2+inlineArgs-1, uint64(999)); err != nil {
		t.Fatalf("Set overflow arg: %v", err)
	}

	after := unsafe.Slice(b.Params(), b.Len())
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("param pointer %d moved after Set", i)
		}
	}
	if got := *(*int32)(after[0]); got != 42 {
		t.Errorf("arg0 = %d, want 42", got)
	}
	if got := unsafe.Slice((*byte)(after[1]), 16); got[0] != 0xB2 {
		t.Errorf("arg1 first byte = %#x, want 0xB2", got[0])
	}
	if got := *(*uint64)(after[len(after)-1]); got != 999 {
		t.Errorf("last arg = %d, want 999", got)
	}
	b.KeepAlive()
}

func TestSetRejects(t *testing.T) {
	var b Builder
	Add(&b, int32(7))
	AddBytes(&b, []byte{1, 2, 3, 4})
	if err := Set(&b, -1, int32(0)); !errors.Is(err, ErrIndexOutOfRange) {
		t.Errorf("Set(-1) = %v, want ErrIndexOutOfRange", err)
	}
	if err := Set(&b, 2, int32(0)); !errors.Is(err, ErrIndexOutOfRange) {
		t.Errorf("Set(len) = %v, want ErrIndexOutOfRange", err)
	}
	if err := Set(&b, 0, float32(1)); !errors.Is(err, ErrTypeMismatch) {
		t.Errorf("Set float32 on int32 = %v, want ErrTypeMismatch", err)
	}
	if err := Set(&b, 0, int64(1)); !errors.Is(err, ErrTypeMismatch) {
		t.Errorf("Set int64 on int32 = %v, want ErrTypeMismatch", err)
	}
	if err := Set(&b, 1, int32(1)); !errors.Is(err, ErrTypeMismatch) {
		t.Errorf("Set on raw slot = %v, want ErrTypeMismatch", err)
	}
}

func TestSetBytes(t *testing.T) {
	var b Builder
	Add(&b, int32(7))
	raw := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	AddBytes(&b, raw)
	if err := b.SetBytes(0, []byte{0xEF, 0xBE, 0xAD, 0xDE}); err != nil {
		t.Fatalf("SetBytes on typed slot: %v", err)
	}
	next := []byte{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	if err := b.SetBytes(1, next); err != nil {
		t.Fatalf("SetBytes on raw slot: %v", err)
	}
	if err := b.SetBytes(1, []byte{1}); !errors.Is(err, ErrSizeMismatch) {
		t.Errorf("SetBytes wrong size = %v, want ErrSizeMismatch", err)
	}
	if err := b.SetBytes(9, next); !errors.Is(err, ErrIndexOutOfRange) {
		t.Errorf("SetBytes bad index = %v, want ErrIndexOutOfRange", err)
	}
	params := unsafe.Slice(b.Params(), b.Len())
	if got := *(*uint32)(params[0]); got != 0xDEADBEEF {
		t.Errorf("arg0 = %#x, want 0xDEADBEEF", got)
	}
	got := unsafe.Slice((*byte)(params[1]), len(next))
	for i := range next {
		if got[i] != next[i] {
			t.Fatalf("raw byte[%d] = %d, want %d", i, got[i], next[i])
		}
	}
	b.KeepAlive()
}

func TestResetReusesArena(t *testing.T) {
	var b Builder
	pack := func() {
		for i := 0; i < inlineArgs+4; i++ {
			Add(&b, uint64(i))
		}
		AddBytes(&b, make([]byte, 32))
		if b.Params() == nil {
			t.Fatal("Params returned nil")
		}
		b.Reset()
	}
	pack()
	if allocs := testing.AllocsPerRun(100, pack); allocs != 0 {
		t.Errorf("repack allocs = %v, want 0", allocs)
	}
}

func TestResetDropsOversizedArena(t *testing.T) {
	var b Builder
	for i := 0; i < 3; i++ {
		AddBytes(&b, make([]byte, 4096))
	}
	if b.Params() == nil {
		t.Fatal("Params returned nil")
	}
	b.Reset()
	if b.spillBuf != nil || b.spillPointers != nil {
		t.Fatal("oversized arena retained after Reset")
	}
	Add(&b, int32(5))
	if got := *(*int32)(unsafe.Slice(b.Params(), 1)[0]); got != 5 {
		t.Fatalf("arg0 after drop = %d, want 5", got)
	}
}

func TestArenaPointerAlignment(t *testing.T) {
	var b Builder
	for i := 0; i < inlineArgs; i++ {
		Add(&b, uint64(i))
	}
	for _, n := range []int{3, 9, 17} {
		AddBytes(&b, make([]byte, n))
	}
	params := unsafe.Slice(b.Params(), b.Len())
	for i := inlineArgs; i < len(params); i++ {
		if uintptr(params[i])%8 != 0 {
			t.Errorf("arena arg %d pointer %% 8 = %d, want 0", i, uintptr(params[i])%8)
		}
	}
	b.KeepAlive()
}

func TestArenaGrowthKeepsValues(t *testing.T) {
	var b Builder
	var want [40][16]byte
	for i := range want {
		want[i][0] = byte(i + 1)
		want[i][15] = byte(0xF0 + i%16)
		Add(&b, want[i])
	}
	params := unsafe.Slice(b.Params(), b.Len())
	for i := range want {
		got := unsafe.Slice((*byte)(params[i]), 16)
		if got[0] != want[i][0] || got[15] != want[i][15] {
			t.Errorf("arg%d = [%#x..%#x], want [%#x..%#x]", i, got[0], got[15], want[i][0], want[i][15])
		}
	}
	b.KeepAlive()
}
