package cudaresult

import (
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// LinkCreate starts a JIT link session and returns its state handle. options and
// values are the parallel CUjit_option / optionValue arrays cuLinkCreate expects;
// the caller owns any buffers the values point at and must keep them alive for
// the whole life of the link state.
func LinkCreate(d *cudasys.Driver, options []int32, values []uintptr) (cudasys.CUlinkState, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuLinkCreate == nil {
		return 0, ErrSymbolUnavailable
	}
	if len(values) != len(options) {
		return 0, ErrInvalidValue
	}
	var optPtr *int32
	var valPtr *uintptr
	if len(options) > 0 {
		optPtr = &options[0]
		valPtr = &values[0]
	}
	var state cudasys.CUlinkState
	if err := check("cuLinkCreate_v2", d.CuLinkCreate(uint32(len(options)), optPtr, valPtr, &state)); err != nil {
		return 0, err
	}
	return state, nil
}

// LinkAddData adds an input image to a link session. inputType is a CUjitInputType
// value; name labels the input in diagnostics and may be nil. data must stay alive
// across the call.
func LinkAddData(d *cudasys.Driver, state cudasys.CUlinkState, inputType uint32, data *byte, size uint64, name *byte) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuLinkAddData == nil {
		return ErrSymbolUnavailable
	}
	return check("cuLinkAddData_v2", d.CuLinkAddData(state, inputType, data, size, name, 0, nil, nil))
}

// LinkComplete finishes linking and returns a pointer to the linked cubin and its
// size. The buffer is owned by the link state and freed by LinkDestroy, so the
// caller must copy it before destroying the state.
func LinkComplete(d *cudasys.Driver, state cudasys.CUlinkState) (unsafe.Pointer, uint64, error) {
	if d == nil {
		return nil, 0, ErrNotInitialized
	}
	if d.CuLinkComplete == nil {
		return nil, 0, ErrSymbolUnavailable
	}
	var ptr unsafe.Pointer
	var size uint64
	if err := check("cuLinkComplete", d.CuLinkComplete(state, &ptr, &size)); err != nil {
		return nil, 0, err
	}
	return ptr, size, nil
}

// LinkDestroy releases a link session previously returned by LinkCreate, along
// with the cubin buffer LinkComplete exposed.
func LinkDestroy(d *cudasys.Driver, state cudasys.CUlinkState) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuLinkDestroy == nil {
		return ErrSymbolUnavailable
	}
	return check("cuLinkDestroy", d.CuLinkDestroy(state))
}
