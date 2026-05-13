//go:build linux

package dynload

import "github.com/ebitengine/purego"

type unixLib struct{ h uintptr }

func (l *unixLib) Handle() uintptr { return l.h }
func (l *unixLib) Close() error    { return purego.Dlclose(l.h) }

type unixOpener struct{}

// Default returns the linux Opener backed by purego.Dlopen.
func Default() Opener { return unixOpener{} }

func (unixOpener) Open(path string) (Library, error) {
	h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, err
	}
	return &unixLib{h: h}, nil
}
