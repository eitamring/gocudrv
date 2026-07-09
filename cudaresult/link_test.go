package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestLinkWrappersNilAndUnavailable(t *testing.T) {
	empty := &cudasys.Driver{}
	cases := []struct {
		name    string
		nilDrv  func() error
		unavail func() error
	}{
		{"create",
			func() error { _, e := LinkCreate(nil, nil, nil); return e },
			func() error { _, e := LinkCreate(empty, nil, nil); return e }},
		{"adddata",
			func() error { return LinkAddData(nil, 0, 1, nil, 0, nil) },
			func() error { return LinkAddData(empty, 0, 1, nil, 0, nil) }},
		{"complete",
			func() error { _, _, e := LinkComplete(nil, 0); return e },
			func() error { _, _, e := LinkComplete(empty, 0); return e }},
		{"destroy",
			func() error { return LinkDestroy(nil, 0) },
			func() error { return LinkDestroy(empty, 0) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if err := c.nilDrv(); !errors.Is(err, ErrNotInitialized) {
				t.Errorf("nil driver = %v, want ErrNotInitialized", err)
			}
			if err := c.unavail(); !errors.Is(err, ErrSymbolUnavailable) {
				t.Errorf("unavailable = %v, want ErrSymbolUnavailable", err)
			}
		})
	}
}

func TestLinkLifecycleWrappers(t *testing.T) {
	cubin := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	var got struct {
		numOptions uint32
		inputType  uint32
		dataSize   uint64
		destroyed  cudasys.CUlinkState
	}
	d := &cudasys.Driver{
		CuLinkCreate: func(n uint32, _ *int32, _ *uintptr, state *cudasys.CUlinkState) cudasys.CUresult {
			got.numOptions = n
			*state = 0x11AA
			return cudasys.CUDA_SUCCESS
		},
		CuLinkAddData: func(state cudasys.CUlinkState, inputType uint32, _ *byte, size uint64, _ *byte, _ uint32, _ *int32, _ *uintptr) cudasys.CUresult {
			if state != 0x11AA {
				t.Errorf("state = %#x, want 0x11AA", state)
			}
			got.inputType = inputType
			got.dataSize = size
			return cudasys.CUDA_SUCCESS
		},
		CuLinkComplete: func(state cudasys.CUlinkState, cubinOut *unsafe.Pointer, sizeOut *uint64) cudasys.CUresult {
			if state != 0x11AA {
				t.Errorf("state = %#x, want 0x11AA", state)
			}
			*cubinOut = unsafe.Pointer(&cubin[0])
			*sizeOut = uint64(len(cubin))
			return cudasys.CUDA_SUCCESS
		},
		CuLinkDestroy: func(state cudasys.CUlinkState) cudasys.CUresult {
			got.destroyed = state
			return cudasys.CUDA_SUCCESS
		},
	}

	opts := []int32{3, 4}
	vals := []uintptr{1, 8192}
	state, err := LinkCreate(d, opts, vals)
	if err != nil {
		t.Fatalf("LinkCreate: %v", err)
	}
	if state != 0x11AA || got.numOptions != 2 {
		t.Errorf("state=%#x numOptions=%d, want 0x11AA, 2", state, got.numOptions)
	}

	if err := LinkAddData(d, state, 1, &cubin[0], 4, nil); err != nil {
		t.Fatalf("LinkAddData: %v", err)
	}
	if got.inputType != 1 || got.dataSize != 4 {
		t.Errorf("inputType=%d size=%d, want 1, 4", got.inputType, got.dataSize)
	}

	ptr, size, err := LinkComplete(d, state)
	if err != nil {
		t.Fatalf("LinkComplete: %v", err)
	}
	if size != uint64(len(cubin)) || ptr != unsafe.Pointer(&cubin[0]) {
		t.Errorf("ptr=%p size=%d, want %p %d", ptr, size, unsafe.Pointer(&cubin[0]), len(cubin))
	}

	if err := LinkDestroy(d, state); err != nil {
		t.Fatalf("LinkDestroy: %v", err)
	}
	if got.destroyed != 0x11AA {
		t.Errorf("destroyed = %#x, want 0x11AA", got.destroyed)
	}
}

func TestLinkCreateLengthMismatch(t *testing.T) {
	d := &cudasys.Driver{
		CuLinkCreate: func(uint32, *int32, *uintptr, *cudasys.CUlinkState) cudasys.CUresult {
			return cudasys.CUDA_SUCCESS
		},
	}
	if _, err := LinkCreate(d, []int32{3, 4}, []uintptr{1}); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("length mismatch = %v, want ErrInvalidValue", err)
	}
}
