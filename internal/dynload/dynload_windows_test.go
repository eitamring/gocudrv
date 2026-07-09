//go:build windows

package dynload

import "testing"

func TestDefaultOpenRealLibrary(t *testing.T) {
	lib, err := Default().Open("kernel32.dll")
	if err != nil {
		t.Fatalf("kernel32.dll not loadable: %v", err)
	}
	if lib.Handle() == 0 {
		t.Error("handle = 0, want non-zero")
	}
	if err := lib.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestDefaultOpenMissingLibrary(t *testing.T) {
	if _, err := Default().Open("gocudrv-does-not-exist.dll"); err == nil {
		t.Fatal("want error for a missing library")
	}
}
