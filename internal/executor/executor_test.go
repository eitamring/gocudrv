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

func BenchmarkDoRoundTrip(b *testing.B) {
	e := New()
	defer func() { _ = e.Close() }()
	fn := func() error { return nil }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := e.Do(fn); err != nil {
			b.Fatal(err)
		}
	}
}

func TestRetireThenClose(t *testing.T) {
	e := New()
	if err := e.Do(func() error { return nil }); err != nil {
		t.Fatalf("Do: %v", err)
	}
	e.Retire()
	if err := e.Close(); err != nil {
		t.Fatalf("Close after Retire: %v", err)
	}
	if err := e.Do(func() error { return nil }); !errors.Is(err, ErrExecutorClosed) {
		t.Errorf("Do after retired Close = %v, want ErrExecutorClosed", err)
	}
}

func TestCloseRunsQueuedTask(t *testing.T) {
	e := New()
	gate := make(chan struct{})
	started := make(chan struct{})
	first := make(chan error, 1)
	second := make(chan error, 1)
	go func() {
		first <- e.Do(func() error { close(started); <-gate; return nil })
	}()
	<-started
	go func() {
		second <- e.Do(func() error { return nil })
	}()
	time.Sleep(20 * time.Millisecond)
	closed := make(chan error, 1)
	go func() { closed <- e.Close() }()
	time.Sleep(20 * time.Millisecond)
	close(gate)
	for name, ch := range map[string]chan error{"first": first, "second": second, "close": closed} {
		select {
		case err := <-ch:
			if err != nil {
				t.Errorf("%s = %v, want nil", name, err)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("%s hung: queued task abandoned on Close", name)
		}
	}
}

func TestDoCtxCancelDuringSpin(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })
	gate := make(chan struct{})
	started := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- e.DoCtx(ctx, func() error { close(started); <-gate; return nil })
	}()
	<-started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("DoCtx = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("DoCtx hung")
	}
	close(gate)
}

type recJob struct {
	gate     chan struct{}
	started  chan struct{}
	recycles atomic.Int32
}

func (j *recJob) Run() error {
	if j.started != nil {
		close(j.started)
	}
	if j.gate != nil {
		<-j.gate
	}
	return nil
}

func (j *recJob) Recycle() { j.recycles.Add(1) }

func waitRecycles(t *testing.T, j *recJob, want int32) {
	t.Helper()
	for i := 0; i < 500; i++ {
		if j.recycles.Load() == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("recycles = %d, want %d", j.recycles.Load(), want)
}

func TestDoJobCtxCancelRecyclesAbandonedJob(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })
	j := &recJob{gate: make(chan struct{}), started: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- e.DoJobCtx(ctx, j) }()
	<-j.started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("DoJobCtx = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("DoJobCtx hung")
	}
	close(j.gate)
	waitRecycles(t, j, 1)
}

func TestRejectedJobIsRecycled(t *testing.T) {
	closed := New()
	if err := closed.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	j := &recJob{}
	if err := closed.DoJob(context.Background(), j); !errors.Is(err, ErrExecutorClosed) {
		t.Fatalf("DoJob on closed = %v, want ErrExecutorClosed", err)
	}
	waitRecycles(t, j, 1)

	e := New()
	t.Cleanup(func() { _ = e.Close() })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	j2 := &recJob{}
	if err := e.DoJobCtx(ctx, j2); !errors.Is(err, context.Canceled) {
		t.Fatalf("DoJobCtx pre-canceled = %v, want context.Canceled", err)
	}
	waitRecycles(t, j2, 1)
}

func TestDrainRecyclesQueuedJob(t *testing.T) {
	e := New()
	gate := make(chan struct{})
	started := make(chan struct{})
	first := make(chan error, 1)
	go func() {
		first <- e.Do(func() error { close(started); <-gate; return nil })
	}()
	<-started
	queued := &recJob{}
	second := make(chan error, 1)
	go func() { second <- e.DoJob(context.Background(), queued) }()
	time.Sleep(20 * time.Millisecond)
	closed := make(chan error, 1)
	go func() { closed <- e.Close() }()
	time.Sleep(20 * time.Millisecond)
	close(gate)
	for name, ch := range map[string]chan error{"first": first, "second": second, "close": closed} {
		select {
		case err := <-ch:
			if err != nil {
				t.Errorf("%s = %v, want nil", name, err)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("%s hung", name)
		}
	}
	waitRecycles(t, queued, 1)
}
