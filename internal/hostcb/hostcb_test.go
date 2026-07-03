package hostcb

import (
	"testing"

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
