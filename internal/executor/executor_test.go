package executor

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m,
		goleak.IgnoreAnyFunction("github.com/eitamring/gocudrv/internal/executor.quarantineThread"),
	)
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

func TestDoCtxNilContext(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })

	if err := e.DoCtx(nil, func() error { return nil }); err != nil { //nolint:staticcheck // Nil is part of the executor contract.
		t.Fatalf("DoCtx with nil context: %v", err)
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

func TestSpinYieldCadence(t *testing.T) {
	for _, tc := range []struct {
		procs int
		want  int
	}{
		{0, 1},
		{1, 1},
		{2, multiPSpinYieldEvery},
		{8, multiPSpinYieldEvery},
	} {
		if got := spinYieldEveryFor(tc.procs); got != tc.want {
			t.Errorf("spinYieldEveryFor(%d) = %d, want %d", tc.procs, got, tc.want)
		}
	}

	cases := []struct {
		spins int
		want  bool
	}{
		{0, false},
		{1, false},
		{multiPSpinYieldEvery - 1, false},
		{multiPSpinYieldEvery, true},
		{multiPSpinYieldEvery + 1, false},
		{2 * multiPSpinYieldEvery, true},
		{callerSpin, true},
	}
	for _, tc := range cases {
		if got := shouldYield(tc.spins, multiPSpinYieldEvery); got != tc.want {
			t.Errorf("shouldYield(%d) = %v, want %v", tc.spins, got, tc.want)
		}
	}
}

func testCompletion() *completion {
	return &completion{result: make(chan error, 1)}
}

func TestCompletionOwnershipOrder(t *testing.T) {
	boom := errors.New("boom")
	t.Run("delivery before receive", func(t *testing.T) {
		c := testCompletion()
		c.result <- boom
		if c.delivered() {
			t.Fatal("worker claimed a completion still owned by the receiver")
		}
		if err := <-c.result; !errors.Is(err, boom) {
			t.Fatalf("result = %v, want boom", err)
		}
		if !c.received() {
			t.Fatal("receiver did not claim delivered completion")
		}
	})

	t.Run("receive before delivery state", func(t *testing.T) {
		c := testCompletion()
		c.result <- boom
		if err := <-c.result; !errors.Is(err, boom) {
			t.Fatalf("result = %v, want boom", err)
		}
		if c.received() {
			t.Fatal("receiver claimed completion before worker published delivery")
		}
		if !c.delivered() {
			t.Fatal("worker did not claim completion after early receive")
		}
	})

	t.Run("abandon before send", func(t *testing.T) {
		c := testCompletion()
		if c.abandon() {
			t.Fatal("caller claimed completion before delivery")
		}
		c.result <- boom
		if !c.delivered() {
			t.Fatal("worker did not claim abandoned completion")
		}
		if len(c.result) != 0 {
			t.Fatal("abandoned result was not drained")
		}
	})

	t.Run("abandon after send before delivery state", func(t *testing.T) {
		c := testCompletion()
		c.result <- boom
		if c.abandon() {
			t.Fatal("caller claimed completion before delivery was published")
		}
		if !c.delivered() {
			t.Fatal("worker did not claim completion abandoned after send")
		}
		if len(c.result) != 0 {
			t.Fatal("abandoned result was not drained")
		}
	})

	t.Run("delivery before abandon", func(t *testing.T) {
		c := testCompletion()
		c.result <- boom
		if c.delivered() {
			t.Fatal("worker claimed completion before caller abandoned it")
		}
		if !c.abandon() {
			t.Fatal("caller did not claim delivered completion")
		}
		if len(c.result) != 0 {
			t.Fatal("abandoned result was not drained")
		}
	})
}

func assertOneCompletionOwner(t *testing.T, owners <-chan bool) {
	t.Helper()
	claimed := 0
	for i := 0; i < 2; i++ {
		if <-owners {
			claimed++
		}
	}
	if claimed != 1 {
		t.Fatalf("completion owners = %d, want 1", claimed)
	}
}

func TestCompletionReceiveRace(t *testing.T) {
	const attempts = 1000
	for i := 0; i < attempts; i++ {
		c := testCompletion()
		owners := make(chan bool, 2)
		start := make(chan struct{})
		go func() {
			<-start
			c.result <- nil
			runtime.Gosched()
			owners <- c.delivered()
		}()
		go func() {
			<-start
			<-c.result
			runtime.Gosched()
			owners <- c.received()
		}()
		close(start)
		assertOneCompletionOwner(t, owners)
		if len(c.result) != 0 {
			t.Fatalf("attempt %d left a result buffered", i)
		}
	}
}

func TestCompletionAbandonRace(t *testing.T) {
	const attempts = 1000
	for i := 0; i < attempts; i++ {
		c := testCompletion()
		owners := make(chan bool, 2)
		start := make(chan struct{})
		go func() {
			<-start
			c.result <- nil
			runtime.Gosched()
			owners <- c.delivered()
		}()
		go func() {
			<-start
			runtime.Gosched()
			owners <- c.abandon()
		}()
		close(start)
		assertOneCompletionOwner(t, owners)
		if len(c.result) != 0 {
			t.Fatalf("attempt %d left a result buffered", i)
		}
	}
}

func TestAbandonedCompletionRecycled(t *testing.T) {
	e := New()
	t.Cleanup(func() { _ = e.Close() })
	started := make(chan struct{})
	finish := make(chan struct{})
	res, err := e.submit(context.Background(), funcJob(func() error {
		close(started)
		<-finish
		return nil
	}))
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	<-started
	if res.abandon() {
		t.Fatal("caller claimed completion before delivery")
	}
	close(finish)

	deadline := time.Now().Add(5 * time.Second)
	for completionState(res.state.Load()) != completionWaiting && time.Now().Before(deadline) {
		runtime.Gosched()
	}
	if got := completionState(res.state.Load()); got != completionWaiting {
		t.Fatalf("completion state = %d, want waiting after recycle", got)
	}
	if len(res.result) != 0 {
		t.Fatal("recycled completion still holds a result")
	}
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
