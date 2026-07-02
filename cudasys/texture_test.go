package cudasys

import (
	"testing"
	"unsafe"
)

// TestTextureStructLayout guards the array and texture descriptor layouts
// against the driver structs on a 64-bit ABI: the driver reads them by offset,
// so a drift would silently corrupt array creation or texture sampling.
func TestTextureStructLayout(t *testing.T) {
	var ad CUDA_ARRAY_DESCRIPTOR
	if got := unsafe.Sizeof(ad); got != 24 {
		t.Errorf("sizeof(CUDA_ARRAY_DESCRIPTOR) = %d, want 24", got)
	}
	if got := unsafe.Offsetof(ad.Format); got != 16 {
		t.Errorf("offsetof(Format) = %d, want 16", got)
	}
	if got := unsafe.Offsetof(ad.NumChannels); got != 20 {
		t.Errorf("offsetof(NumChannels) = %d, want 20", got)
	}

	var rd CUDA_RESOURCE_DESC
	if got := unsafe.Sizeof(rd); got != 144 {
		t.Errorf("sizeof(CUDA_RESOURCE_DESC) = %d, want 144", got)
	}
	if got := unsafe.Offsetof(rd.Handle); got != 8 {
		t.Errorf("offsetof(Handle) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(rd.Flags); got != 136 {
		t.Errorf("offsetof(Flags) = %d, want 136", got)
	}

	var td CUDA_TEXTURE_DESC
	if got := unsafe.Sizeof(td); got != 104 {
		t.Errorf("sizeof(CUDA_TEXTURE_DESC) = %d, want 104", got)
	}
	checks := []struct {
		name      string
		got, want uintptr
	}{
		{"FilterMode", unsafe.Offsetof(td.FilterMode), 12},
		{"Flags", unsafe.Offsetof(td.Flags), 16},
		{"MipmapFilterMode", unsafe.Offsetof(td.MipmapFilterMode), 24},
		{"MipmapLevelBias", unsafe.Offsetof(td.MipmapLevelBias), 28},
		{"BorderColor", unsafe.Offsetof(td.BorderColor), 40},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("offsetof(%s) = %d, want %d", c.name, c.got, c.want)
		}
	}
}
