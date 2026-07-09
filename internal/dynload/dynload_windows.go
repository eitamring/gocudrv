//go:build windows

package dynload

import (
	"syscall"
	"unsafe"
)

// loadLibrarySearchSystem32 is LOAD_LIBRARY_SEARCH_SYSTEM32: a bare library
// name resolves from %windir%\System32 only, never the application directory
// or PATH, which blocks DLL preloading. The driver installs nvcuda.dll there.
const loadLibrarySearchSystem32 = 0x00000800

var (
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	procLoadLibraryExW = kernel32.NewProc("LoadLibraryExW")
)

type winLib struct{ h syscall.Handle }

func (l *winLib) Handle() uintptr { return uintptr(l.h) }
func (l *winLib) Close() error    { return syscall.FreeLibrary(l.h) }

type winOpener struct{}

// Default returns the windows Opener backed by LoadLibraryExW with the
// search path restricted to System32.
func Default() Opener { return winOpener{} }

func (winOpener) Open(path string) (Library, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	h, _, callErr := procLoadLibraryExW.Call(uintptr(unsafe.Pointer(p)), 0, loadLibrarySearchSystem32)
	if h == 0 {
		return nil, callErr
	}
	return &winLib{h: syscall.Handle(h)}, nil
}
