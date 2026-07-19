//go:build !windows

package cudasys

import "github.com/eitamring/gocudrv/internal/dynload"

// bindByValueIPC binds the two IPC entry points whose 64-byte handle argument
// passes by value. Off windows the struct passes per the platform C ABI,
// which SyscallN cannot express portably, so these keep the registered binding.
func bindByValueIPC(lib dynload.Library, d *Driver) {
	if _, err := lookupFn(lib, "cuIpcOpenMemHandle_v2"); err == nil {
		_ = bind(lib, &d.CuIpcOpenMemHandle, "cuIpcOpenMemHandle_v2")
	}
	if _, err := lookupFn(lib, "cuIpcOpenEventHandle"); err == nil {
		_ = bind(lib, &d.CuIpcOpenEventHandle, "cuIpcOpenEventHandle")
	}
}
