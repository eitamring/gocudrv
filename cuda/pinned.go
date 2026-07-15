package cuda

import "sync"

// PinnedHost is a typed view of page-locked host memory that can be used by
// CUDA copy operations. HostBuffer and RegisteredHost are its only
// implementations.
type PinnedHost[T Supported] interface {
	Slice() []T
	Len() int
	Bytes() uint64
	pinnedHost() pinnedHostRef[T]
}

type pinnedHostRef[T Supported] struct {
	ctx    *Context
	ptr    *byte
	length int
	lock   *sync.RWMutex
	closed *bool
	owner  any
}

func pinnedHostRefOf[T Supported](host PinnedHost[T]) (pinnedHostRef[T], error) {
	if host == nil {
		return pinnedHostRef[T]{}, ErrNilBuffer
	}
	var ref pinnedHostRef[T]
	switch host := host.(type) {
	case *HostBuffer[T]:
		ref = host.pinnedHost()
	case *RegisteredHost[T]:
		ref = host.pinnedHost()
	}
	if ref.owner == nil || ref.lock == nil || ref.closed == nil {
		return pinnedHostRef[T]{}, ErrNilBuffer
	}
	return ref, nil
}
