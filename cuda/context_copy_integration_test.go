//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func BenchmarkRealBlockingCopyCommandLatency(b *testing.B) {
	ctx := benchRealContext(b)
	buf, err := Alloc[byte](ctx, 1)
	if err != nil {
		b.Fatalf("Alloc: %v", err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	stream, err := ctx.NewStream()
	if err != nil {
		b.Fatalf("NewStream: %v", err)
	}
	b.Cleanup(func() { _ = stream.Close() })
	dst := make([]byte, 1)
	if err := buf.CopyTo(context.Background(), dst); err != nil {
		b.Fatalf("warm copy: %v", err)
	}
	defaultStream := &Stream{ctx: ctx}

	b.ResetTimer()
	var commandTime time.Duration
	for range b.N {
		entered := make(chan struct{})
		release := make(chan struct{})
		var releaseOnce sync.Once
		unblock := func() { releaseOnce.Do(func() { close(release) }) }
		err := defaultStream.LaunchHostFunc(func() {
			close(entered)
			<-release
		})
		if errors.Is(err, ErrSymbolUnavailable) {
			b.Skip("driver lacks cuLaunchHostFunc")
		}
		if err != nil {
			b.Fatalf("LaunchHostFunc: %v", err)
		}
		select {
		case <-entered:
		case <-time.After(10 * time.Second):
			unblock()
			b.Fatal("host function did not start")
		}

		copyDone := make(chan error, 1)
		go func() { copyDone <- buf.CopyTo(context.Background(), dst) }()
		time.Sleep(time.Millisecond)
		safety := time.AfterFunc(100*time.Millisecond, unblock)
		start := time.Now()
		commandErr := stream.Query()
		commandTime += time.Since(start)
		unblock()
		safety.Stop()
		if commandErr != nil {
			b.Fatalf("Query: %v", commandErr)
		}
		if err := <-copyDone; err != nil {
			b.Fatalf("CopyTo: %v", err)
		}
	}
	b.ReportMetric(float64(commandTime.Microseconds())/float64(b.N), "command-us/op")
}
