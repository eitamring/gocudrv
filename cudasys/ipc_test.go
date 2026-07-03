package cudasys

import (
	"testing"
	"unsafe"
)

// TestIpcHandleLayout guards the IPC handle layouts: the driver copies exactly
// CU_IPC_HANDLE_SIZE (64) bytes through them, by value.
func TestIpcHandleLayout(t *testing.T) {
	var m CUipcMemHandle
	if got := unsafe.Sizeof(m); got != 64 {
		t.Errorf("sizeof(CUipcMemHandle) = %d, want 64", got)
	}
	if got := unsafe.Offsetof(m.Data); got != 0 {
		t.Errorf("offsetof(Data) = %d, want 0", got)
	}
	var e CUipcEventHandle
	if got := unsafe.Sizeof(e); got != 64 {
		t.Errorf("sizeof(CUipcEventHandle) = %d, want 64", got)
	}
}
