package cuda

import (
	"context"

	"github.com/eitamring/gocudrv/cudasys"
)

// View is a non-owning window into a region of a Buffer's device memory. It
// carries an element offset and length into the owning Buffer and is produced
// by Buffer.View or, for a sub-window, View.View.
//
// A View does not own memory: it has no Close, and the underlying allocation is
// freed only when the owning Buffer is closed. Copy operations on a View run
// through the owner, so once the owner is closed they return ErrBufferClosed.
// Keep the owning Buffer open for as long as any View of it is in use.
type View[T Supported] struct {
	owner  *Buffer[T]
	offset int
	length int
}

// View returns a non-owning view of n elements of b starting at element offset.
// It returns ErrInvalidLength for a negative offset or non-positive n, and
// ErrOutOfRange if the range does not fit the buffer. The buffer must be open.
func (b *Buffer[T]) View(offset, n int) (*View[T], error) {
	if b == nil {
		return nil, ErrNilBuffer
	}
	b.opMu.RLock()
	defer b.opMu.RUnlock()
	if b.closed {
		return nil, ErrBufferClosed
	}
	if offset < 0 || n <= 0 {
		return nil, ErrInvalidLength
	}
	if offset > b.length-n {
		return nil, ErrOutOfRange
	}
	return &View[T]{owner: b, offset: offset, length: n}, nil
}

// View returns a sub-view of n elements of v starting at element offset, with
// the same validation as Buffer.View. The sub-view references the same owning
// Buffer.
func (v *View[T]) View(offset, n int) (*View[T], error) {
	if v == nil || v.owner == nil {
		return nil, ErrNilBuffer
	}
	if offset < 0 || n <= 0 {
		return nil, ErrInvalidLength
	}
	if offset > v.length-n {
		return nil, ErrOutOfRange
	}
	return &View[T]{owner: v.owner, offset: v.offset + offset, length: n}, nil
}

// Len is the number of elements the view spans. It is 0 for a nil view.
func (v *View[T]) Len() int {
	if v == nil {
		return 0
	}
	return v.length
}

// Bytes is the size of the view in bytes. It is 0 for a nil view.
func (v *View[T]) Bytes() uint64 {
	if v == nil {
		return 0
	}
	return uint64(v.length) * elemSize[T]()
}

// DevicePtr is the device pointer at the start of the view. Like
// Buffer.DevicePtr it is a raw snapshot, valid only while the owning Buffer is
// open, and is zero for a nil view.
func (v *View[T]) DevicePtr() cudasys.CUdeviceptr {
	if v == nil || v.owner == nil {
		return 0
	}
	return v.owner.offsetPtr(v.offset)
}

// CopyFrom copies the host slice into the view; len(src) must equal the view
// length. It returns ErrBufferClosed once the owning Buffer has been closed.
func (v *View[T]) CopyFrom(ctx context.Context, src []T) error {
	if v == nil || v.owner == nil {
		return ErrNilBuffer
	}
	if len(src) != v.length {
		return ErrLengthMismatch
	}
	return v.owner.CopyFromAt(ctx, v.offset, src)
}

// CopyTo copies the view into the host slice; len(dst) must equal the view
// length. It returns ErrBufferClosed once the owning Buffer has been closed.
func (v *View[T]) CopyTo(ctx context.Context, dst []T) error {
	if v == nil || v.owner == nil {
		return ErrNilBuffer
	}
	if len(dst) != v.length {
		return ErrLengthMismatch
	}
	return v.owner.CopyToAt(ctx, dst, v.offset)
}
