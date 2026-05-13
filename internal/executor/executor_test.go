package executor

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestDo(t *testing.T) {
	boom := errors.New("boom")
	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil error", func() error { return nil }, nil},
		{"explicit error", func() error { return boom }, boom},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := New()
			t.Cleanup(func() { _ = e.Close() })
			if err := e.Do(tc.fn); !errors.Is(err, tc.want) && err != tc.want {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestDoPanicRecovered(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	err := e.Do(func() error { panic("kaboom") })
	var pe *PanicError
	if !errors.As(err, &pe) {
		t.Fatalf("err = %T %v, want *PanicError", err, err)
	}
	if pe.Value != "kaboom" {
		t.Errorf("value = %v, want %q", pe.Value, "kaboom")
	}
	if !errors.Is(err, &PanicError{}) {
		t.Error("errors.Is against zero PanicError did not match")
	}

	if err := e.Do(func() error { return nil }); err != nil {
		t.Errorf("executor unusable after panic: %v", err)
	}
}

func TestConcurrentDoSerializes(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	var counter int
	var wg sync.WaitGroup
	const n = 200
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			_ = e.Do(func() error {
				counter++
				return nil
			})
		}()
	}
	wg.Wait()
	if counter != n {
		t.Errorf("counter = %d, want %d (lost increment implies non-serialized execution)", counter, n)
	}
}

func TestCloseIdempotent(t *testing.T) {
	e := New()
	if err := e.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Errorf("second close: %v", err)
	}
}

func TestDoAfterClose(t *testing.T) {
	e := New()
	if err := e.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := e.Do(func() error { return nil }); !errors.Is(err, ErrExecutorClosed) {
		t.Errorf("err = %v, want ErrExecutorClosed", err)
	}
}

func TestCloseWaitsForSubmittedWork(t *testing.T) {
	e := New()

	started := make(chan struct{})
	finish := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- e.Do(func() error {
			close(started)
			<-finish
			return nil
		})
	}()

	<-started
	closeDone := make(chan error, 1)
	go func() { closeDone <- e.Close() }()

	select {
	case err := <-closeDone:
		t.Fatalf("Close returned before submitted work finished: %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	close(finish)
	if err := <-done; err != nil {
		t.Fatalf("submitted work: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := e.Do(func() error { return nil }); !errors.Is(err, ErrExecutorClosed) {
		t.Errorf("Do after Close err = %v, want ErrExecutorClosed", err)
	}
}

func TestDoCtxCanceledBeforeSubmit(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := e.DoCtx(ctx, func() error {
		t.Error("fn must not run when ctx is canceled before submit")
		return nil
	}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestDoCtxCanceledMidExecution(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	start := make(chan struct{})
	finish := make(chan struct{})
	ran := atomic.Bool{}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-start
		cancel()
	}()

	err := e.DoCtx(ctx, func() error {
		close(start)
		<-finish
		ran.Store(true)
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	close(finish)

	// Give the executor a moment to finish the abandoned task, then submit
	// another to confirm it is still healthy.
	deadline := time.Now().Add(time.Second)
	for !ran.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !ran.Load() {
		t.Error("fn did not complete on executor after ctx cancel")
	}
	if err := e.Do(func() error { return nil }); err != nil {
		t.Errorf("executor unusable after canceled task: %v", err)
	}
}

func TestSingleWorkerGoroutine(t *testing.T) {
	// If Do ran on multiple goroutines, two concurrent Do calls could
	// observe the counter incremented twice in the window between read and
	// write. With a single worker, increments are sequential.
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	var inside atomic.Int32
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			_ = e.Do(func() error {
				if inside.Add(1) != 1 {
					t.Error("more than one task inside executor at once")
				}
				inside.Add(-1)
				return nil
			})
		}()
	}
	wg.Wait()
}
