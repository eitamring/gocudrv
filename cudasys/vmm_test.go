package cudasys

import (
	"testing"
	"unsafe"
)

// TestVMMStructLayout guards the VMM struct layouts against their CUDA C
// counterparts on a 64-bit ABI: the driver reads them by offset, so a drift
// would corrupt every virtual-memory call.
func TestVMMStructLayout(t *testing.T) {
	sizes := []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"CUmemLocation", unsafe.Sizeof(CUmemLocation{}), 8},
		{"CUmemAllocationFlags", unsafe.Sizeof(CUmemAllocationFlags{}), 8},
		{"CUmemAllocationProp", unsafe.Sizeof(CUmemAllocationProp{}), 32},
		{"CUmemAccessDesc", unsafe.Sizeof(CUmemAccessDesc{}), 12},
	}
	for _, s := range sizes {
		if s.got != s.want {
			t.Errorf("sizeof(%s) = %d, want %d", s.name, s.got, s.want)
		}
	}

	var p CUmemAllocationProp
	offsets := []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"RequestedHandleTypes", unsafe.Offsetof(p.RequestedHandleTypes), 4},
		{"Location", unsafe.Offsetof(p.Location), 8},
		{"Win32HandleMetaData", unsafe.Offsetof(p.Win32HandleMetaData), 16},
		{"AllocFlags", unsafe.Offsetof(p.AllocFlags), 24},
	}
	for _, o := range offsets {
		if o.got != o.want {
			t.Errorf("offsetof(CUmemAllocationProp.%s) = %d, want %d", o.name, o.got, o.want)
		}
	}

	var a CUmemAccessDesc
	if got := unsafe.Offsetof(a.Flags); got != 8 {
		t.Errorf("offsetof(CUmemAccessDesc.Flags) = %d, want 8", got)
	}

	var loc CUmemLocation
	if got := unsafe.Offsetof(loc.Id); got != 4 {
		t.Errorf("offsetof(CUmemLocation.Id) = %d, want 4", got)
	}
	var fl CUmemAllocationFlags
	if got := unsafe.Offsetof(fl.Usage); got != 2 {
		t.Errorf("offsetof(CUmemAllocationFlags.Usage) = %d, want 2", got)
	}
	if got := unsafe.Offsetof(fl.Reserved); got != 4 {
		t.Errorf("offsetof(CUmemAllocationFlags.Reserved) = %d, want 4", got)
	}
}
