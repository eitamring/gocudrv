package cudasys

import (
	"testing"
	"unsafe"
)

// TestMemcpy2DLayout guards the Memcpy2D layout against CUDA_MEMCPY2D on a
// 64-bit ABI: the driver reads the struct by offset, so a drift here would
// silently corrupt every 2D copy.
func TestMemcpy2DLayout(t *testing.T) {
	var m Memcpy2D
	if got := unsafe.Sizeof(m); got != 128 {
		t.Errorf("sizeof(Memcpy2D) = %d, want 128", got)
	}
	checks := []struct {
		name      string
		got, want uintptr
	}{
		{"SrcMemoryType", unsafe.Offsetof(m.SrcMemoryType), 16},
		{"SrcHost", unsafe.Offsetof(m.SrcHost), 24},
		{"SrcDevice", unsafe.Offsetof(m.SrcDevice), 32},
		{"SrcPitch", unsafe.Offsetof(m.SrcPitch), 48},
		{"DstXInBytes", unsafe.Offsetof(m.DstXInBytes), 56},
		{"DstHost", unsafe.Offsetof(m.DstHost), 80},
		{"WidthInBytes", unsafe.Offsetof(m.WidthInBytes), 112},
		{"Height", unsafe.Offsetof(m.Height), 120},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("offsetof(%s) = %d, want %d", c.name, c.got, c.want)
		}
	}
}
