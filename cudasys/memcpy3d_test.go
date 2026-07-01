package cudasys

import (
	"testing"
	"unsafe"
)

// TestMemcpy3DLayout guards the Memcpy3D layout against CUDA_MEMCPY3D on a
// 64-bit ABI: the driver reads the struct by offset, so a drift here would
// silently corrupt every 3D copy.
func TestMemcpy3DLayout(t *testing.T) {
	var m Memcpy3D
	if got := unsafe.Sizeof(m); got != 200 {
		t.Errorf("sizeof(Memcpy3D) = %d, want 200", got)
	}
	checks := []struct {
		name      string
		got, want uintptr
	}{
		{"SrcXInBytes", unsafe.Offsetof(m.SrcXInBytes), 0},
		{"SrcZ", unsafe.Offsetof(m.SrcZ), 16},
		{"SrcMemoryType", unsafe.Offsetof(m.SrcMemoryType), 32},
		{"SrcHost", unsafe.Offsetof(m.SrcHost), 40},
		{"SrcDevice", unsafe.Offsetof(m.SrcDevice), 48},
		{"SrcArray", unsafe.Offsetof(m.SrcArray), 56},
		{"SrcPitch", unsafe.Offsetof(m.SrcPitch), 72},
		{"SrcHeight", unsafe.Offsetof(m.SrcHeight), 80},
		{"DstXInBytes", unsafe.Offsetof(m.DstXInBytes), 88},
		{"DstMemoryType", unsafe.Offsetof(m.DstMemoryType), 120},
		{"DstHost", unsafe.Offsetof(m.DstHost), 128},
		{"DstDevice", unsafe.Offsetof(m.DstDevice), 136},
		{"DstArray", unsafe.Offsetof(m.DstArray), 144},
		{"DstPitch", unsafe.Offsetof(m.DstPitch), 160},
		{"DstHeight", unsafe.Offsetof(m.DstHeight), 168},
		{"WidthInBytes", unsafe.Offsetof(m.WidthInBytes), 176},
		{"Height", unsafe.Offsetof(m.Height), 184},
		{"Depth", unsafe.Offsetof(m.Depth), 192},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("offsetof(%s) = %d, want %d", c.name, c.got, c.want)
		}
	}
}
