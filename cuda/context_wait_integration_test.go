//go:build cuda_integration

package cuda

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestRealBlockedSynchronizeLeavesCommandsAndSetupFree(t *testing.T) {
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

	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	err = stream.LaunchHostFunc(func() {
		close(entered)
		<-release
	})
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks cuLaunchHostFunc")
	}
	if err != nil {
		t.Fatalf("LaunchHostFunc: %v", err)
	}

	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("host function did not start")
	}
	syncDone := make(chan error, 1)
	go func() { syncDone <- stream.Synchronize(context.Background()) }()
	time.Sleep(20 * time.Millisecond)
	commandDone := make(chan error, 1)
	go func() {
		_, _, err := ctx.MemInfo()
		commandDone <- err
	}()
	select {
	case err := <-commandDone:
		if err != nil {
			t.Fatalf("MemInfo: %v", err)
		}
	case <-time.After(2 * time.Second):
		unblock()
		t.Fatal("MemInfo blocked behind stream synchronization")
	}
	type eventResult struct {
		event *Event
		err   error
	}
	eventDone := make(chan eventResult, 1)
	go func() {
		event, err := ctx.NewEvent()
		eventDone <- eventResult{event: event, err: err}
	}()
	var event *Event
	select {
	case result := <-eventDone:
		if result.err != nil {
			t.Fatalf("NewEvent: %v", result.err)
		}
		event = result.event
	case <-time.After(2 * time.Second):
		unblock()
		t.Fatal("NewEvent blocked behind stream synchronization")
	}
	unblock()
	t.Cleanup(func() { _ = event.Close() })
	select {
	case err := <-syncDone:
		if err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Synchronize did not finish")
	}
	if err := event.Close(); err != nil {
		t.Fatalf("Event.Close: %v", err)
	}
}

func waitForActiveSyncs(tb testing.TB, ctx *Context, want int64) {
	tb.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for ctx.syncActive.Load() < want {
		if time.Now().After(deadline) {
			tb.Fatalf("tracked waits = %d, want at least %d", ctx.syncActive.Load(), want)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestRealBlockedStreamWaitLeavesOtherWaitFree(t *testing.T) {
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
	streamA, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream A: %v", err)
	}
	t.Cleanup(func() { _ = streamA.Close() })
	streamB, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream B: %v", err)
	}
	t.Cleanup(func() { _ = streamB.Close() })

	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	err = streamA.LaunchHostFunc(func() {
		close(entered)
		<-release
	})
	if errors.Is(err, ErrSymbolUnavailable) {
		t.Skip("driver lacks cuLaunchHostFunc")
	}
	if err != nil {
		t.Fatalf("LaunchHostFunc: %v", err)
	}
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("host function did not start")
	}

	syncA := make(chan error, 1)
	go func() { syncA <- streamA.Synchronize(context.Background()) }()
	waitForActiveSyncs(t, ctx, 1)

	syncB := make(chan error, 1)
	go func() { syncB <- streamB.Synchronize(context.Background()) }()
	select {
	case err := <-syncB:
		if err != nil {
			t.Fatalf("Synchronize B: %v", err)
		}
	case <-time.After(2 * time.Second):
		unblock()
		t.Fatal("stream B synchronize blocked behind stream A")
	}
	select {
	case err := <-syncA:
		unblock()
		t.Fatalf("stream A synchronize completed early: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	unblock()
	select {
	case err := <-syncA:
		if err != nil {
			t.Fatalf("Synchronize A: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("stream A synchronize did not finish")
	}
}
