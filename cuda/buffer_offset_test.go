package cuda

import (
	"context"
	"errors"
	"testing"
)

const offsetBase = 0x10000

func newOffsetContext(t *testing.T, calls *memCalls) *Context {
	t.Helper()
	return newTestContext(t, fakeMemoryDriver(calls, offsetBase))
}

func TestCopyFromAt(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	if err := buf.CopyFromAt(context.Background(), 4, make([]float32, 8)); err != nil {
		t.Fatalf("CopyFromAt: %v", err)
	}
	if calls.htod.Load() != 1 {
		t.Errorf("htod calls = %d, want 1", calls.htod.Load())
	}
	if got, want := calls.lastDst.Load(), uintptr(offsetBase+4*4); got != want {
		t.Errorf("dst ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSize.Load(), uint64(8*4); got != want {
		t.Errorf("bytes = %d, want %d", got, want)
	}
}

func TestCopyToAt(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	if err := buf.CopyToAt(context.Background(), make([]float32, 8), 4); err != nil {
		t.Fatalf("CopyToAt: %v", err)
	}
	if calls.dtoh.Load() != 1 {
		t.Errorf("dtoh calls = %d, want 1", calls.dtoh.Load())
	}
	if got, want := calls.lastSrc.Load(), uintptr(offsetBase+4*4); got != want {
		t.Errorf("src ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSize.Load(), uint64(8*4); got != want {
		t.Errorf("bytes = %d, want %d", got, want)
	}
}

func TestCopyToDeviceAt(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	src, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc src: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })
	dst, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	if err := src.CopyToDeviceAt(context.Background(), 2, dst, 5, 6); err != nil {
		t.Fatalf("CopyToDeviceAt: %v", err)
	}
	if calls.dtod.Load() != 1 {
		t.Errorf("dtod calls = %d, want 1", calls.dtod.Load())
	}
	if got, want := calls.lastDst.Load(), uintptr(offsetBase+2*4); got != want {
		t.Errorf("dst ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSrc.Load(), uintptr(offsetBase+5*4); got != want {
		t.Errorf("src ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSize.Load(), uint64(6*4); got != want {
		t.Errorf("bytes = %d, want %d", got, want)
	}
}

func TestCopyToDeviceAtAsync(t *testing.T) {
	var calls memCalls
	ctx, stream, _, _ := newAsyncCopyFixture(t, &calls)
	src, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc src: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })
	dst, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	if err := src.CopyToDeviceAtAsync(context.Background(), stream, 1, dst, 3, 4); err != nil {
		t.Fatalf("CopyToDeviceAtAsync: %v", err)
	}
	if calls.dtodAsync.Load() != 1 {
		t.Errorf("dtodAsync calls = %d, want 1", calls.dtodAsync.Load())
	}
	if got, want := calls.lastDst.Load(), uintptr(0xDEAD+1*4); got != want {
		t.Errorf("dst ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSrc.Load(), uintptr(0xDEAD+3*4); got != want {
		t.Errorf("src ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSize.Load(), uint64(4*4); got != want {
		t.Errorf("bytes = %d, want %d", got, want)
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}
}

// TestOffsetCopyRejects checks that out-of-range and invalid arguments are
// rejected before any CUDA call (the driver copy counters stay at zero).
func TestOffsetCopyRejects(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	dst, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	bg := context.Background()
	cases := []struct {
		name    string
		run     func() error
		wantErr error
	}{
		{"from negative offset", func() error { return buf.CopyFromAt(bg, -1, make([]float32, 4)) }, ErrInvalidLength},
		{"from empty src", func() error { return buf.CopyFromAt(bg, 0, nil) }, ErrInvalidLength},
		{"from overruns", func() error { return buf.CopyFromAt(bg, 12, make([]float32, 8)) }, ErrOutOfRange},
		{"to negative offset", func() error { return buf.CopyToAt(bg, make([]float32, 4), -1) }, ErrInvalidLength},
		{"to overruns", func() error { return buf.CopyToAt(bg, make([]float32, 8), 12) }, ErrOutOfRange},
		{"dtod zero count", func() error { return buf.CopyToDeviceAt(bg, 0, dst, 0, 0) }, ErrInvalidLength},
		{"dtod src overruns", func() error { return buf.CopyToDeviceAt(bg, 0, dst, 13, 8) }, ErrOutOfRange},
		{"dtod dst overruns", func() error { return buf.CopyToDeviceAt(bg, 13, dst, 0, 8) }, ErrOutOfRange},
		{"dtod context mismatch", func() error {
			other := &Buffer[float32]{ctx: &Context{}, length: 16}
			return buf.CopyToDeviceAt(bg, 0, other, 0, 4)
		}, ErrContextMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
	if calls.htod.Load() != 0 || calls.dtoh.Load() != 0 || calls.dtod.Load() != 0 {
		t.Errorf("a rejected copy reached the driver: htod=%d dtoh=%d dtod=%d",
			calls.htod.Load(), calls.dtoh.Load(), calls.dtod.Load())
	}
}

// TestCopyToDeviceAtSelfCopy exercises the dst==b alias path: an in-place
// subrange copy must not take a second read lock on the buffer (which can
// deadlock against a concurrent Close).
func TestCopyToDeviceAtSelfCopy(t *testing.T) {
	var calls memCalls
	ctx := newOffsetContext(t, &calls)
	buf, err := Alloc[float32](ctx, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })

	if err := buf.CopyToDeviceAt(context.Background(), 0, buf, 8, 4); err != nil {
		t.Fatalf("self CopyToDeviceAt: %v", err)
	}
	if calls.dtod.Load() != 1 {
		t.Errorf("dtod = %d, want 1", calls.dtod.Load())
	}
	if got, want := calls.lastSrc.Load(), uintptr(offsetBase+8*4); got != want {
		t.Errorf("src = %#x, want %#x", got, want)
	}
	if got, want := calls.lastDst.Load(), uintptr(offsetBase+0); got != want {
		t.Errorf("dst = %#x, want %#x", got, want)
	}
}
