package cudasys

import (
	"testing"
	"unsafe"
)

// TestArray3DDescriptorLayout guards the layout against CUDA_ARRAY3D_DESCRIPTOR
// on a 64-bit ABI: the driver reads it by offset, so a drift would silently
// corrupt array creation.
func TestArray3DDescriptorLayout(t *testing.T) {
	var d CUDA_ARRAY3D_DESCRIPTOR
	if got := unsafe.Sizeof(d); got != 40 {
		t.Errorf("sizeof(CUDA_ARRAY3D_DESCRIPTOR) = %d, want 40", got)
	}
	checks := []struct {
		name      string
		got, want uintptr
	}{
		{"Width", unsafe.Offsetof(d.Width), 0},
		{"Height", unsafe.Offsetof(d.Height), 8},
		{"Depth", unsafe.Offsetof(d.Depth), 16},
		{"Format", unsafe.Offsetof(d.Format), 24},
		{"NumChannels", unsafe.Offsetof(d.NumChannels), 28},
		{"Flags", unsafe.Offsetof(d.Flags), 32},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("offsetof(%s) = %d, want %d", c.name, c.got, c.want)
		}
	}
}
