//go:build windows

package cudasys

import (
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/eitamring/gocudrv/internal/dynload"
)

// bindByValueIPC binds the two IPC entry points whose 64-byte handle argument
// passes by value. The win64 ABI passes aggregates larger than 8 bytes by
// reference to a caller-owned copy, so the adapters pass a pointer to the
// local copy the Go call already made.
func bindByValueIPC(lib dynload.Library, d *Driver) {
	if addr, err := lookupFn(lib, "cuIpcOpenMemHandle_v2"); err == nil {
		d.CuIpcOpenMemHandle = func(pdptr *CUdeviceptr, handle CUipcMemHandle, flags uint32) CUresult {
			return cures(purego.SyscallN(addr, uintptr(unsafe.Pointer(pdptr)), uintptr(unsafe.Pointer(&handle)), uintptr(flags)))
		}
	}
	if addr, err := lookupFn(lib, "cuIpcOpenEventHandle"); err == nil {
		d.CuIpcOpenEventHandle = func(phEvent *CUevent, handle CUipcEventHandle) CUresult {
			return cures(purego.SyscallN(addr, uintptr(unsafe.Pointer(phEvent)), uintptr(unsafe.Pointer(&handle))))
		}
	}
}
