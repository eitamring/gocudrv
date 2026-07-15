package hostcb

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ebitengine/purego"
)

func TestRegisterInvokeLifecycle(t *testing.T) {
	ran := false
	k := Register(func() { ran = true })
	if !Pending(k) {
		t.Fatal("key not pending after Register")
	}
	Invoke(k)
	if !ran {
		t.Fatal("closure did not run")
	}
	if Pending(k) {
		t.Fatal("key still pending after Invoke")
	}
	Invoke(k)
}

func TestConcurrentLifecycle(t *testing.T) {
	const workers = 16
	const perWorker = 200
	var ran atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				key := Register(func() { ran.Add(1) })
				if i%2 == 0 {
					Invoke(key)
					Invoke(key)
				} else {
					Unregister(key)
				}
				if Pending(key) {
					t.Errorf("key %d is still pending", key)
				}
			}
		}()
	}
	wg.Wait()
	if want := int64(workers * perWorker / 2); ran.Load() != want {
		t.Errorf("callbacks = %d, want %d", ran.Load(), want)
	}
}

func TestConcurrentInvokeOnce(t *testing.T) {
	var ran atomic.Int64
	key := Register(func() { ran.Add(1) })
	var wg sync.WaitGroup
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			Invoke(key)
		}()
	}
	wg.Wait()
	if ran.Load() != 1 {
		t.Errorf("callbacks = %d, want 1", ran.Load())
	}
	if Pending(key) {
		t.Fatal("key is still pending")
	}
}

func TestInvokeUnregisterRace(t *testing.T) {
	const count = 1000
	var ran atomic.Int64
	for range count {
		key := Register(func() { ran.Add(1) })
		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			Invoke(key)
		}()
		go func() {
			defer wg.Done()
			<-start
			Unregister(key)
		}()
		close(start)
		wg.Wait()
		if Pending(key) {
			t.Fatalf("key %d is still pending", key)
		}
	}
	if got := ran.Load(); got > count {
		t.Errorf("callbacks = %d, want at most %d", got, count)
	}
}

func TestInvokeCanRegister(t *testing.T) {
	done := make(chan struct{})
	key := Register(func() {
		inner := Register(func() {})
		Unregister(inner)
		close(done)
	})
	go Invoke(key)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("callback registry deadlocked")
	}
}

func BenchmarkRegisterInvoke(b *testing.B) {
	fn := func() {}
	b.ReportAllocs()
	for b.Loop() {
		Invoke(Register(fn))
	}
}

func BenchmarkCallback(b *testing.B) {
	Callback()
	b.ReportAllocs()
	for b.Loop() {
		_ = Callback()
	}
}

func BenchmarkRegisterInvokeParallel(b *testing.B) {
	fn := func() {}
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Invoke(Register(fn))
		}
	})
}

func TestUnregister(t *testing.T) {
	k := Register(func() { t.Fatal("must not run") })
	Unregister(k)
	Invoke(k)
}

func TestInvokeSwallowsPanic(t *testing.T) {
	k := Register(func() { panic("boom") })
	Invoke(k)
	if Pending(k) {
		t.Fatal("panicking closure not removed")
	}
}

func TestCallbackRoundTripThroughCPointer(t *testing.T) {
	ptr := Callback()
	if ptr == 0 {
		t.Fatal("nil trampoline")
	}
	if Callback() != ptr {
		t.Fatal("trampoline not cached")
	}
	got := make(chan uintptr, 1)
	k := Register(func() { got <- 1 })
	purego.SyscallN(ptr, k)
	select {
	case <-got:
	default:
		t.Fatal("C-pointer round trip did not run the closure")
	}
}
