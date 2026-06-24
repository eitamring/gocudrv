package cuda

import (
	"context"
	"errors"
	"testing"
)

func TestViewLenBytesPtr(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	v, err := buf.View(4, 8)
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	if v.Len() != 8 {
		t.Errorf("Len = %d, want 8", v.Len())
	}
	if v.Bytes() != 8*4 {
		t.Errorf("Bytes = %d, want 32", v.Bytes())
	}
	if got, want := v.DevicePtr(), buf.DevicePtr()+4*4; got != want {
		t.Errorf("DevicePtr = %#x, want %#x", got, want)
	}
}

func TestViewCopies(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	v, err := buf.View(4, 8)
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	bg := context.Background()

	if err := v.CopyFrom(bg, make([]float32, 8)); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}
	if calls.htod.Load() != 1 {
		t.Errorf("htod = %d, want 1", calls.htod.Load())
	}
	if got, want := calls.lastDst.Load(), uintptr(offsetBase+4*4); got != want {
		t.Errorf("htod dst = %#x, want %#x", got, want)
	}
	if calls.lastSize.Load() != 8*4 {
		t.Errorf("htod bytes = %d, want 32", calls.lastSize.Load())
	}

	if err := v.CopyTo(bg, make([]float32, 8)); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	if got, want := calls.lastSrc.Load(), uintptr(offsetBase+4*4); got != want {
		t.Errorf("dtoh src = %#x, want %#x", got, want)
	}

	// Length must match the view exactly.
	if err := v.CopyFrom(bg, make([]float32, 7)); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short CopyFrom = %v, want ErrLengthMismatch", err)
	}
}

func TestSubView(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	v, err := buf.View(4, 8) // elements 4..12
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	sub, err := v.View(2, 3) // elements 6..9
	if err != nil {
		t.Fatalf("sub View: %v", err)
	}
	if sub.Len() != 3 {
		t.Errorf("sub Len = %d, want 3", sub.Len())
	}
	if got, want := sub.DevicePtr(), buf.DevicePtr()+6*4; got != want {
		t.Errorf("sub DevicePtr = %#x, want %#x", got, want)
	}
	if _, err := v.View(2, 10); !errors.Is(err, ErrOutOfRange) {
		t.Errorf("oversized sub-view = %v, want ErrOutOfRange", err)
	}
}

func TestViewRejects(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	cases := []struct {
		name      string
		offset, n int
		wantErr   error
	}{
		{"negative offset", -1, 4, ErrInvalidLength},
		{"zero length", 0, 0, ErrInvalidLength},
		{"overruns", 12, 8, ErrOutOfRange},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := buf.View(tc.offset, tc.n); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestViewClosedOwner(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	v, err := buf.View(0, 4)
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	if err := buf.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	bg := context.Background()
	if err := v.CopyFrom(bg, make([]float32, 4)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyFrom after owner close = %v, want ErrBufferClosed", err)
	}
	if err := v.CopyTo(bg, make([]float32, 4)); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("CopyTo after owner close = %v, want ErrBufferClosed", err)
	}
	if _, err := buf.View(0, 4); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("View on closed buffer = %v, want ErrBufferClosed", err)
	}
	if _, err := v.View(0, 2); !errors.Is(err, ErrBufferClosed) {
		t.Errorf("sub-view of closed owner = %v, want ErrBufferClosed", err)
	}
	// The copies must never have reached the driver.
	if calls.htod.Load() != 0 || calls.dtoh.Load() != 0 {
		t.Errorf("closed-owner copy reached driver: htod=%d dtoh=%d", calls.htod.Load(), calls.dtoh.Load())
	}
}

func TestViewNilReceiver(t *testing.T) {
	var v *View[float32]
	if v.Len() != 0 {
		t.Error("nil Len should be 0")
	}
	if v.Bytes() != 0 {
		t.Error("nil Bytes should be 0")
	}
	if v.DevicePtr() != 0 {
		t.Error("nil DevicePtr should be 0")
	}
	if err := v.CopyFrom(context.Background(), make([]float32, 4)); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil CopyFrom = %v, want ErrNilBuffer", err)
	}
	if _, err := v.View(0, 1); !errors.Is(err, ErrNilBuffer) {
		t.Errorf("nil sub-view = %v, want ErrNilBuffer", err)
	}
}
