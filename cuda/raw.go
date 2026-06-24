package cuda

import "github.com/eitamring/gocudrv/cudasys"

// Raw handle accessors expose the underlying CUDA driver handles that gocudrv
// normally keeps private. They exist so a sibling module (for example a cuBLAS,
// cuDNN, or TensorRT binding) can share this package's context, streams, events,
// and device buffers instead of creating its own.
//
// These accessors bypass the safety guarantees the typed API provides:
//
//   - No lifetime tracking. A handle is valid only while the Go value that owns
//     it is open. Using a handle after the owner's Close is undefined behavior,
//     and gocudrv cannot detect it.
//   - No thread affinity. CUDA's current context is per OS thread. gocudrv runs
//     context-affine work on a pinned executor goroutine; any raw driver call you
//     make yourself must ensure the right context is current on the calling
//     thread.
//   - No locking. The returned handle is a plain snapshot; the typed API's lock
//     does not protect work you issue through it.
//
// Prefer the typed methods. Reach for these only when integrating with code that
// must be handed a raw CUDA handle.

// Raw returns the underlying primary CUcontext. It is zero on a nil receiver and
// is valid only until the Context is closed.
func (c *Context) Raw() cudasys.CUcontext {
	if c == nil {
		return 0
	}
	return c.raw
}

// Driver returns the loaded CUDA driver with every bound entry point, so a
// sibling module can issue driver calls through the library gocudrv already
// opened instead of loading libcuda a second time. It is nil on a nil receiver.
func (c *Context) Driver() *cudasys.Driver {
	if c == nil {
		return nil
	}
	return c.driver
}

// Raw returns the underlying CUstream. It is zero on a nil receiver and is valid
// only until the Stream is closed.
func (s *Stream) Raw() cudasys.CUstream {
	if s == nil {
		return 0
	}
	return s.raw
}

// Raw returns the underlying CUevent. It is zero on a nil receiver and is valid
// only until the Event is closed.
func (e *Event) Raw() cudasys.CUevent {
	if e == nil {
		return 0
	}
	return e.raw
}

// DevicePtr returns the raw device pointer for the buffer's memory, for passing
// to code that takes a CUdeviceptr directly. It is zero on a nil receiver and is
// valid only until the Buffer is closed or freed with FreeAsync. Use Len and
// Bytes for the element count and byte size.
func (b *Buffer[T]) DevicePtr() cudasys.CUdeviceptr {
	if b == nil {
		return 0
	}
	return b.ptr
}
