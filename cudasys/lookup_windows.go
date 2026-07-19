//go:build windows

package cudasys

import (
	"fmt"
	"syscall"

	"github.com/eitamring/gocudrv/internal/dynload"
)

// lookupSymbol resolves a driver entry point to its raw address with
// GetProcAddress.
func lookupSymbol(lib dynload.Library, name string) (uintptr, error) {
	addr, err := syscall.GetProcAddress(syscall.Handle(lib.Handle()), name)
	if err != nil {
		return 0, fmt.Errorf("cudasys: lookup %q: %w", name, err)
	}
	if addr == 0 {
		return 0, fmt.Errorf("cudasys: lookup %q: nil address", name)
	}
	return addr, nil
}
