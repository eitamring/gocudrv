//go:build windows

package cudasys

import "github.com/eitamring/gocudrv/internal/dynload"

// bindByValueIPC binds the two IPC entry points whose 64-byte handle argument
// passes by value. The win64 ABI passes aggregates larger than 8 bytes by
// reference to a caller-owned copy, and purego does not marshal struct values
// on windows, so each is bound with the pointer signature the ABI actually
// uses and adapted to the Driver's by-value field.
func bindByValueIPC(lib dynload.Library, d *Driver) {
	var openMem func(pdptr *CUdeviceptr, handle *CUipcMemHandle, flags uint32) CUresult
	if bindFn(lib, &openMem, "cuIpcOpenMemHandle_v2") == nil {
		d.CuIpcOpenMemHandle = func(pdptr *CUdeviceptr, handle CUipcMemHandle, flags uint32) CUresult {
			return openMem(pdptr, &handle, flags)
		}
	}
	var openEvent func(phEvent *CUevent, handle *CUipcEventHandle) CUresult
	if bindFn(lib, &openEvent, "cuIpcOpenEventHandle") == nil {
		d.CuIpcOpenEventHandle = func(phEvent *CUevent, handle CUipcEventHandle) CUresult {
			return openEvent(phEvent, &handle)
		}
	}
}
