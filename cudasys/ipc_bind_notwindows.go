//go:build !windows

package cudasys

import "github.com/eitamring/gocudrv/internal/dynload"

// bindByValueIPC binds the two IPC entry points whose 64-byte handle argument
// passes by value. On SysV targets purego places the struct on the stack
// directly, so the plain signatures bind best-effort like any other optional.
func bindByValueIPC(lib dynload.Library, d *Driver) {
	_ = bindFn(lib, &d.CuIpcOpenMemHandle, "cuIpcOpenMemHandle_v2")
	_ = bindFn(lib, &d.CuIpcOpenEventHandle, "cuIpcOpenEventHandle")
}
