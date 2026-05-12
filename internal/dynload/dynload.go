package dynload

import (
	"errors"
	"fmt"
)

// Library is a loaded shared-library handle. Handle returns the OS-level
// handle used by ABI dispatchers such as purego.
type Library interface {
	Handle() uintptr
	Close() error
}

// Opener opens a shared library by path. Implementations are OS-specific.
type Opener interface {
	Open(path string) (Library, error)
}

// Sentinel errors returned by the loader.
var (
	ErrUnsupported  = errors.New("dynload: platform not supported")
	ErrNoCandidates = errors.New("dynload: no library candidates provided")
)

// OpenAny tries each path with o in order and returns the first successful
// Library. If every path fails, the returned error joins all individual
// open errors with the path that produced each one.
func OpenAny(o Opener, paths []string) (Library, error) {
	if len(paths) == 0 {
		return nil, ErrNoCandidates
	}
	errs := make([]error, 0, len(paths))
	for _, p := range paths {
		lib, err := o.Open(p)
		if err == nil {
			return lib, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", p, err))
	}
	return nil, errors.Join(errs...)
}
