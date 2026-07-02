//go:build linux

package dynload

import "testing"

func TestDefaultOpenRealLibrary(t *testing.T) {
	lib, err := Default().Open("libc.so.6")
	if err != nil {
		t.Skipf("libc.so.6 not loadable here: %v", err)
	}
	if lib.Handle() == 0 {
		t.Error("handle = 0, want non-zero")
	}
	if err := lib.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestDefaultOpenMissingLibrary(t *testing.T) {
	if _, err := Default().Open("libgocudrv-does-not-exist.so.999"); err == nil {
		t.Fatal("want error for a missing library")
	}
}
