//go:build windows

package dynload

import "syscall"

type winLib struct{ h syscall.Handle }

func (l *winLib) Handle() uintptr { return uintptr(l.h) }
func (l *winLib) Close() error    { return syscall.FreeLibrary(l.h) }

type winOpener struct{}

// Default returns the windows Opener backed by syscall.LoadLibrary.
func Default() Opener { return winOpener{} }

func (winOpener) Open(path string) (Library, error) {
	h, err := syscall.LoadLibrary(path)
	if err != nil {
		return nil, err
	}
	return &winLib{h: h}, nil
}
