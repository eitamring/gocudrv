//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"testing"
)

func realPinnedAsyncFixture(t *testing.T) (*Context, *Stream) {
	t.Helper()
	initOrSkip(t)
	dev, err := GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatalf("Primary: %v", err)
	}
	t.Cleanup(func() { _ = ctx.Close() })
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	return ctx, stream
}

func requirePinnedAsync(t *testing.T, operation string, err error) {
	t.Helper()
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skipf("%s is unavailable: %v", operation, err)
	}
	if err != nil {
		t.Fatalf("%s: %v", operation, err)
	}
}

func checkPinnedAsyncValues(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRealRegisteredHostAsyncRoundTrip(t *testing.T) {
	ctx, stream := realPinnedAsyncFixture(t)
	const n = 257
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*1.25 + 0.5
	}
	dst := make([]float32, n)
	srcHost, err := RegisterHost(ctx, src)
	requirePinnedAsync(t, "RegisterHost source", err)
	t.Cleanup(func() { _ = srcHost.Close() })
	dstHost, err := RegisterHost(ctx, dst)
	requirePinnedAsync(t, "RegisterHost destination", err)
	t.Cleanup(func() { _ = dstHost.Close() })
	buf, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	bg := context.Background()
	t.Cleanup(func() { _ = stream.Synchronize(bg) })
	requirePinnedAsync(t, "CopyFromHostAsync", buf.CopyFromHostAsync(bg, stream, srcHost))
	requirePinnedAsync(t, "CopyToHostAsync", buf.CopyToHostAsync(bg, stream, dstHost))
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	checkPinnedAsyncValues(t, dst, src)
}

func TestRealPitchedPinnedAsyncRoundTrip(t *testing.T) {
	ctx, stream := realPinnedAsyncFixture(t)
	const width, height = 37, 11
	const n = width * height
	hostIn, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost input: %v", err)
	}
	t.Cleanup(func() { _ = hostIn.Close() })
	hostOut, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost output: %v", err)
	}
	t.Cleanup(func() { _ = hostOut.Close() })
	input := hostIn.Slice()
	for i := range input {
		input[i] = float32(i)*0.75 - 2
	}
	src, err := AllocPitched[float32](ctx, width, height)
	requirePinnedAsync(t, "AllocPitched source", err)
	t.Cleanup(func() { _ = src.Close() })
	dst, err := AllocPitched[float32](ctx, width, height)
	requirePinnedAsync(t, "AllocPitched destination", err)
	t.Cleanup(func() { _ = dst.Close() })
	bg := context.Background()
	t.Cleanup(func() { _ = stream.Synchronize(bg) })
	requirePinnedAsync(t, "Pitched CopyFromHostAsync", src.CopyFromHostAsync(bg, stream, hostIn))
	requirePinnedAsync(t, "Pitched CopyToDeviceAsync", src.CopyToDeviceAsync(bg, stream, dst))
	requirePinnedAsync(t, "Pitched CopyToHostAsync", dst.CopyToHostAsync(bg, stream, hostOut))
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	checkPinnedAsyncValues(t, hostOut.Slice(), input)
}

func TestRealVolumePinnedAsyncRoundTrip(t *testing.T) {
	ctx, stream := realPinnedAsyncFixture(t)
	const width, height, depth = 19, 7, 3
	const n = width * height * depth
	hostIn, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost input: %v", err)
	}
	t.Cleanup(func() { _ = hostIn.Close() })
	hostOut, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost output: %v", err)
	}
	t.Cleanup(func() { _ = hostOut.Close() })
	input := hostIn.Slice()
	for i := range input {
		input[i] = float32(i)*2.5 + 3
	}
	volume, err := AllocVolume[float32](ctx, width, height, depth)
	requirePinnedAsync(t, "AllocVolume", err)
	t.Cleanup(func() { _ = volume.Close() })
	bg := context.Background()
	t.Cleanup(func() { _ = stream.Synchronize(bg) })
	requirePinnedAsync(t, "Volume CopyFromHostAsync", volume.CopyFromHostAsync(bg, stream, hostIn))
	requirePinnedAsync(t, "Volume CopyToHostAsync", volume.CopyToHostAsync(bg, stream, hostOut))
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	checkPinnedAsyncValues(t, hostOut.Slice(), input)
}

func TestRealArrayPinnedAsyncRoundTrip(t *testing.T) {
	ctx, stream := realPinnedAsyncFixture(t)
	const width, height = 29, 13
	const n = width * height
	hostIn, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost input: %v", err)
	}
	t.Cleanup(func() { _ = hostIn.Close() })
	hostOut, err := AllocHost[float32](ctx, n)
	if err != nil {
		t.Fatalf("AllocHost output: %v", err)
	}
	t.Cleanup(func() { _ = hostOut.Close() })
	input := hostIn.Slice()
	for i := range input {
		input[i] = float32(i)*0.125 + 7
	}
	array, err := AllocArray2D[float32](ctx, width, height)
	requirePinnedAsync(t, "AllocArray2D", err)
	t.Cleanup(func() { _ = array.Close() })
	bg := context.Background()
	t.Cleanup(func() { _ = stream.Synchronize(bg) })
	requirePinnedAsync(t, "Array CopyFromHostAsync", array.CopyFromHostAsync(bg, stream, hostIn))
	requirePinnedAsync(t, "Array CopyToHostAsync", array.CopyToHostAsync(bg, stream, hostOut))
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	checkPinnedAsyncValues(t, hostOut.Slice(), input)
}
