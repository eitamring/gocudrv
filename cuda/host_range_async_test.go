package cuda

import (
	"context"
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// newRangeAsyncFixture is newAsyncCopyFixture sized for a genuine sub-range:
// n elements each on the device buffer and the pinned host buffer, large
// enough that offset+n < n on both sides for the happy-path test below.
func newRangeAsyncFixture(t *testing.T, calls *memCalls, n int) (*Context, *Stream, *Buffer[float32], *HostBuffer[float32]) {
	t.Helper()
	drv := fakeMemoryDriver(calls, 0xDEAD)
	drv.CuMemAllocHost = func(pp **byte, bytes uint64) cudasys.CUresult {
		storage := make([]byte, int(bytes))
		*pp = &storage[0]
		return cudasys.CUDA_SUCCESS
	}
	drv.CuMemFreeHost = func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuStreamCreate = func(stream *cudasys.CUstream, _ uint32) cudasys.CUresult {
		*stream = 0x5151
		return cudasys.CUDA_SUCCESS
	}
	drv.CuStreamDestroy = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	drv.CuStreamSynchronize = func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }

	ctx := newTestContext(t, drv)
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	buf, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	host, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = host.Close() })
	return ctx, stream, buf, host
}

// TestCopyFromHostRangeAsync checks that a sub-range copy on BOTH sides
// (mismatched src/dst offsets, so the two are not accidentally symmetric)
// lands the correct device offset, the correct host BYTES (content, not
// just pointer arithmetic — an offset error that reads the wrong host
// elements would still land at a plausible-looking pointer), the correct
// byte count, and the correct stream — the exact shape goinfer's expert-slot
// DMA needs: a stacked pinned host buffer's expert e into a device slot s,
// where e and s are unrelated indices.
func TestCopyFromHostRangeAsync(t *testing.T) {
	var calls memCalls
	var captured []byte
	_, stream, buf, host := newRangeAsyncFixture(t, &calls, 16)
	src := host.Slice()
	for i := range src {
		src[i] = float32(i + 1) // 1..16, so a wrong offset reads a wrong value, not a zero
	}
	buf.ctx.driver.CuMemcpyHtoDAsync = func(dst cudasys.CUdeviceptr, srcPtr *byte, bytes uint64, stream cudasys.CUstream) cudasys.CUresult {
		calls.htodAsync.Add(1)
		calls.lastDst.Store(uintptr(dst))
		calls.lastSize.Store(bytes)
		calls.lastStream.Store(uintptr(stream))
		captured = append([]byte(nil), unsafe.Slice(srcPtr, bytes)...)
		return cudasys.CUDA_SUCCESS
	}

	const dstOff, srcOff, n = 3, 7, 6 // dst [3:9), src [7:13) of a 16-element buffer/host
	if err := buf.CopyFromHostRangeAsync(context.Background(), stream, dstOff, host, srcOff, n); err != nil {
		t.Fatalf("CopyFromHostRangeAsync: %v", err)
	}
	if calls.htodAsync.Load() != 1 {
		t.Errorf("async htod calls = %d, want 1", calls.htodAsync.Load())
	}
	if got, want := calls.lastDst.Load(), uintptr(0xDEAD+dstOff*4); got != want {
		t.Errorf("dst ptr = %#x, want %#x", got, want)
	}
	if got, want := calls.lastSize.Load(), uint64(n*4); got != want {
		t.Errorf("bytes = %d, want %d", got, want)
	}
	if calls.lastStream.Load() != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", calls.lastStream.Load())
	}
	want := unsafe.Slice((*byte)(unsafe.Pointer(&src[srcOff])), n*4)
	for i := range want {
		if captured[i] != want[i] {
			t.Fatalf("captured[%d] = %d, want %d (host offset not applied correctly)", i, captured[i], want[i])
		}
	}
}

// TestCopyFromHostRangeAsyncRejects checks that out-of-range and invalid
// arguments are rejected before any CUDA call, mirroring
// TestOffsetCopyRejects' shape for the new offset+async+pinned combination.
func TestCopyFromHostRangeAsyncRejects(t *testing.T) {
	var calls memCalls
	ctx, stream, buf, host := newRangeAsyncFixture(t, &calls, 16)
	bg := context.Background()

	closedStream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	if err := closedStream.Close(); err != nil {
		t.Fatalf("Stream.Close: %v", err)
	}

	otherDrv := fakeMemoryDriver(&memCalls{}, 0xF00D)
	otherDrv.CuMemAllocHost = func(pp **byte, bytes uint64) cudasys.CUresult {
		storage := make([]byte, int(bytes))
		*pp = &storage[0]
		return cudasys.CUDA_SUCCESS
	}
	otherDrv.CuMemFreeHost = func(*byte) cudasys.CUresult { return cudasys.CUDA_SUCCESS }
	otherCtx := newTestContext(t, otherDrv)
	otherHost, err := AllocHost[float32](otherCtx, 16)
	if err != nil {
		t.Fatalf("AllocHost (other ctx): %v", err)
	}
	t.Cleanup(func() { _ = otherHost.Close() })

	cases := []struct {
		name    string
		run     func() error
		wantErr error
	}{
		{"nil stream", func() error { return buf.CopyFromHostRangeAsync(bg, nil, 0, host, 0, 4) }, ErrNilStream},
		{"closed stream", func() error { return buf.CopyFromHostRangeAsync(bg, closedStream, 0, host, 0, 4) }, ErrStreamClosed},
		{"negative dst offset", func() error { return buf.CopyFromHostRangeAsync(bg, stream, -1, host, 0, 4) }, ErrInvalidLength},
		{"negative src offset", func() error { return buf.CopyFromHostRangeAsync(bg, stream, 0, host, -1, 4) }, ErrInvalidLength},
		{"zero n", func() error { return buf.CopyFromHostRangeAsync(bg, stream, 0, host, 0, 0) }, ErrInvalidLength},
		{"dst overruns", func() error { return buf.CopyFromHostRangeAsync(bg, stream, 13, host, 0, 8) }, ErrOutOfRange},
		{"src overruns", func() error { return buf.CopyFromHostRangeAsync(bg, stream, 0, host, 13, 8) }, ErrOutOfRange},
		{"host context mismatch", func() error { return buf.CopyFromHostRangeAsync(bg, stream, 0, otherHost, 0, 4) }, ErrContextMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
	if calls.htodAsync.Load() != 0 {
		t.Errorf("a rejected copy reached the driver: htodAsync=%d", calls.htodAsync.Load())
	}
}
