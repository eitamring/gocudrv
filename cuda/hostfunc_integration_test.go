//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestRealLaunchHostFunc(t *testing.T) {
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

	const n = 1 << 20
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
	src := host.Slice()
	for i := range src {
		src[i] = float32(i % 97)
	}

	var mu sync.Mutex
	var order []int
	signal := make(chan struct{}, 2)
	record := func(k int) func() {
		return func() {
			mu.Lock()
			order = append(order, k)
			mu.Unlock()
			signal <- struct{}{}
		}
	}

	bg := context.Background()
	if err := buf.CopyFromHostAsync(bg, stream, host); err != nil {
		t.Fatalf("CopyFromHostAsync: %v", err)
	}
	err = stream.LaunchHostFunc(record(1))
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks cuLaunchHostFunc")
	}
	if err != nil {
		t.Fatalf("LaunchHostFunc 1: %v", err)
	}
	if err := stream.LaunchHostFunc(record(2)); err != nil {
		t.Fatalf("LaunchHostFunc 2: %v", err)
	}

	for i := 0; i < 2; i++ {
		select {
		case <-signal:
		case <-time.After(10 * time.Second):
			t.Fatalf("host function %d never ran", i+1)
		}
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(order) != 2 || order[0] != 1 || order[1] != 2 {
		t.Fatalf("order = %v, want [1 2] (stream order)", order)
	}
	t.Log("two host functions ran in stream order on the driver's thread")
}
