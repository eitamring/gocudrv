//go:build !linux && !windows

package dynload

type unsupportedOpener struct{}

// Default returns an Opener that always reports ErrUnsupported.
func Default() Opener { return unsupportedOpener{} }

func (unsupportedOpener) Open(string) (Library, error) {
	return nil, ErrUnsupported
}
