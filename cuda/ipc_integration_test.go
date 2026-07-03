//go:build cuda_integration

package cuda

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"testing"
)

const ipcUnsupportedMarker = "GOCUDRV_IPC_UNSUPPORTED"

func ipcUnsupported(err error) bool {
	return errors.Is(err, ErrSymbolUnavailable) || errors.Is(err, ErrNotSupported) ||
		errors.Is(err, ErrNotPermitted) || errors.Is(err, ErrOperatingSystem)
}

// TestRealIPCMemRoundTrip shares a device buffer with a real child process:
// the parent fills it and exports the handle, the child maps it, verifies the
// contents, and writes a reply pattern the parent checks.
func TestRealIPCMemRoundTrip(t *testing.T) {
	if os.Getenv("GOCUDRV_IPC_CHILD") == "1" {
		t.Skip("parent test; the child runs TestRealIPCChild")
	}
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

	const n = 256
	buf, err := Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = buf.Close() })
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 2
	}
	bg := context.Background()
	if err := buf.CopyFrom(bg, src); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	h, err := buf.IPCHandle()
	if ipcUnsupported(err) {
		t.Skipf("IPC not supported here: %v", err)
	}
	if err != nil {
		t.Fatalf("IPCHandle: %v", err)
	}
	raw := h.Bytes()

	var eventEnv string
	ev, err := ctx.NewEvent(WithEventInterprocess())
	if err == nil {
		t.Cleanup(func() { _ = ev.Close() })
		if eh, err := ev.IPCHandle(); err == nil {
			raw2 := eh.Bytes()
			eventEnv = hex.EncodeToString(raw2[:])
		} else if !ipcUnsupported(err) {
			t.Fatalf("event IPCHandle: %v", err)
		}
	} else if !ipcUnsupported(err) {
		t.Fatalf("NewEvent interprocess: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run", "TestRealIPCChild$", "-test.v")
	cmd.Env = append(os.Environ(),
		"GOCUDRV_IPC_CHILD=1",
		"GOCUDRV_IPC_HANDLE="+hex.EncodeToString(raw[:]),
		"GOCUDRV_IPC_EVENT_HANDLE="+eventEnv,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child process failed: %v\n%s", err, out)
	}
	if bytes.Contains(out, []byte(ipcUnsupportedMarker)) {
		t.Skipf("child reports IPC import unsupported on this platform:\n%s", out)
	}

	got := make([]float32, n)
	if err := buf.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range got {
		if want := float32(i)*3 + 1; got[i] != want {
			t.Fatalf("element %d = %v, want %v (child reply)", i, got[i], want)
		}
	}
	t.Logf("child mapped the buffer, verified %d elements, and replied; event handle shared: %v", n, eventEnv != "")
}

// TestRealIPCChild is the child half of TestRealIPCMemRoundTrip; it only runs
// when the parent spawns it with the handle in the environment.
func TestRealIPCChild(t *testing.T) {
	if os.Getenv("GOCUDRV_IPC_CHILD") != "1" {
		t.Skip("helper; spawned by TestRealIPCMemRoundTrip")
	}
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

	rawSlice, err := hex.DecodeString(os.Getenv("GOCUDRV_IPC_HANDLE"))
	if err != nil || len(rawSlice) != 64 {
		t.Fatalf("bad handle env: %v (%d bytes)", err, len(rawSlice))
	}
	var raw [64]byte
	copy(raw[:], rawSlice)

	const n = 256
	imp, err := OpenIPCBuffer[float32](ctx, IPCMemHandleFromBytes(raw), n)
	if ipcUnsupported(err) {
		t.Log(ipcUnsupportedMarker)
		t.Skipf("IPC import unsupported: %v", err)
	}
	if err != nil {
		t.Fatalf("OpenIPCBuffer: %v", err)
	}
	t.Cleanup(func() { _ = imp.Close() })

	bg := context.Background()
	got := make([]float32, n)
	if err := imp.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i := range got {
		if want := float32(i) * 2; got[i] != want {
			t.Fatalf("element %d = %v, want %v (parent data)", i, got[i], want)
		}
	}
	reply := make([]float32, n)
	for i := range reply {
		reply[i] = float32(i)*3 + 1
	}
	if err := imp.CopyFrom(bg, reply); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	if evHex := os.Getenv("GOCUDRV_IPC_EVENT_HANDLE"); evHex != "" {
		evSlice, err := hex.DecodeString(evHex)
		if err != nil || len(evSlice) != 64 {
			t.Fatalf("bad event handle env: %v", err)
		}
		var evRaw [64]byte
		copy(evRaw[:], evSlice)
		ev, err := OpenIPCEvent(ctx, IPCEventHandleFromBytes(evRaw))
		if ipcUnsupported(err) {
			t.Log(ipcUnsupportedMarker)
			t.Skipf("IPC event import unsupported: %v", err)
		}
		if err != nil {
			t.Fatalf("OpenIPCEvent: %v", err)
		}
		if err := ev.Close(); err != nil {
			t.Fatalf("imported event Close: %v", err)
		}
	}
}
