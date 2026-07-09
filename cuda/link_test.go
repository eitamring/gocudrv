package cuda

import (
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

type linkFake struct {
	createCalls     atomic.Int32
	addCalls        atomic.Int32
	completeCalls   atomic.Int32
	destroyCalls    atomic.Int32
	lastInputType   atomic.Uint32
	destroyed       atomic.Bool
	addAfterDestroy atomic.Bool
	failDestroys    atomic.Int32
	lastOptions     []int32
	lastAddData     []byte
	lastName        []byte
	cubin           []byte
}

func linkBaseDriver() *cudasys.Driver {
	return &cudasys.Driver{
		CuDeviceGetCount: func(n *int32) cudasys.CUresult { *n = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRetain: func(ctx *cudasys.CUcontext, _ cudasys.CUdevice) cudasys.CUresult {
			*ctx = 0xC0FFEE
			return cudasys.CUDA_SUCCESS
		},
		CuDevicePrimaryCtxRelease: func(cudasys.CUdevice) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		CuCtxSetCurrent:           func(cudasys.CUcontext) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
	}
}

func linkDriver(f *linkFake) *cudasys.Driver {
	d := linkBaseDriver()
	d.CuLinkCreate = func(n uint32, options *int32, _ *uintptr, state *cudasys.CUlinkState) cudasys.CUresult {
		f.createCalls.Add(1)
		f.lastOptions = append([]int32(nil), unsafe.Slice(options, n)...)
		*state = 0x11AA
		return cudasys.CUDA_SUCCESS
	}
	d.CuLinkAddData = func(_ cudasys.CUlinkState, inputType uint32, data *byte, size uint64, name *byte, _ uint32, _ *int32, _ *uintptr) cudasys.CUresult {
		f.addCalls.Add(1)
		if f.destroyed.Load() {
			f.addAfterDestroy.Store(true)
		}
		f.lastInputType.Store(inputType)
		f.lastAddData = append([]byte(nil), unsafe.Slice(data, size)...)
		if name != nil {
			length := 0
			for {
				b := *(*byte)(unsafe.Add(unsafe.Pointer(name), length))
				length++
				if b == 0 {
					break
				}
			}
			f.lastName = append([]byte(nil), unsafe.Slice(name, length)...)
		} else {
			f.lastName = nil
		}
		return cudasys.CUDA_SUCCESS
	}
	d.CuLinkComplete = func(_ cudasys.CUlinkState, cubinOut *unsafe.Pointer, sizeOut *uint64) cudasys.CUresult {
		f.completeCalls.Add(1)
		*cubinOut = unsafe.Pointer(&f.cubin[0])
		*sizeOut = uint64(len(f.cubin))
		return cudasys.CUDA_SUCCESS
	}
	d.CuLinkDestroy = func(cudasys.CUlinkState) cudasys.CUresult {
		f.destroyCalls.Add(1)
		if f.failDestroys.Load() > 0 {
			f.failDestroys.Add(-1)
			return cudasys.CUDA_ERROR_INVALID_VALUE
		}
		f.destroyed.Store(true)
		return cudasys.CUDA_SUCCESS
	}
	return d
}

func TestLinkerFlow(t *testing.T) {
	f := &linkFake{cubin: []byte{0xDE, 0xAD, 0xBE, 0xEF}}
	ctx := newTestContext(t, linkDriver(f))

	lk, err := ctx.NewLinker(JITOptions{LogBufferBytes: 256, MaxRegisters: 32})
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	if f.createCalls.Load() != 1 {
		t.Errorf("create calls = %d, want 1", f.createCalls.Load())
	}
	wantOpts := []int32{jitInfoLogBuffer, jitInfoLogBufferSizeBytes, jitErrorLogBuffer, jitErrorLogBufferSizeBytes, jitMaxRegisters}
	if len(f.lastOptions) != len(wantOpts) {
		t.Fatalf("options = %v, want %v", f.lastOptions, wantOpts)
	}
	for i := range wantOpts {
		if f.lastOptions[i] != wantOpts[i] {
			t.Errorf("option[%d] = %d, want %d", i, f.lastOptions[i], wantOpts[i])
		}
	}

	ptx := []byte("some ptx")
	if err := lk.AddPTX("k.ptx", ptx); err != nil {
		t.Fatalf("AddPTX: %v", err)
	}
	if f.lastInputType.Load() != jitInputPTX {
		t.Errorf("input type = %d, want %d (PTX)", f.lastInputType.Load(), jitInputPTX)
	}
	if len(f.lastAddData) != len(ptx)+1 || f.lastAddData[len(ptx)] != 0 {
		t.Errorf("AddPTX data = %v, want %d bytes ending in NUL", f.lastAddData, len(ptx)+1)
	}
	wantName := []byte("k.ptx\x00")
	if string(f.lastName) != string(wantName) {
		t.Errorf("name = %q, want %q", f.lastName, wantName)
	}

	cubin, err := lk.Complete()
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if string(cubin) != string(f.cubin) {
		t.Errorf("cubin = %v, want %v", cubin, f.cubin)
	}
	f.cubin[0] = 0x00
	if cubin[0] != 0xDE {
		t.Error("Complete returned an alias of the driver-owned buffer, not a copy")
	}

	if err := lk.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := lk.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if f.destroyCalls.Load() != 1 {
		t.Errorf("destroy calls = %d, want 1", f.destroyCalls.Load())
	}
}

func TestLinkerAddCubin(t *testing.T) {
	f := &linkFake{cubin: []byte{1, 2, 3}}
	ctx := newTestContext(t, linkDriver(f))
	lk, err := ctx.NewLinker(JITOptions{})
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	t.Cleanup(func() { _ = lk.Close() })

	cubin := []byte{0xAA, 0xBB, 0xCC}
	if err := lk.AddCubin("", cubin); err != nil {
		t.Fatalf("AddCubin: %v", err)
	}
	if f.lastInputType.Load() != jitInputCubin {
		t.Errorf("input type = %d, want %d (cubin)", f.lastInputType.Load(), jitInputCubin)
	}
	if len(f.lastAddData) != len(cubin) {
		t.Errorf("AddCubin passed %d bytes, want %d (exact, no NUL)", len(f.lastAddData), len(cubin))
	}
	if f.lastName != nil {
		t.Errorf("empty name should pass nil, got %v", f.lastName)
	}
}

func TestLinkerRejects(t *testing.T) {
	f := &linkFake{cubin: []byte{1}}
	ctx := newTestContext(t, linkDriver(f))
	lk, err := ctx.NewLinker(JITOptions{})
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	t.Cleanup(func() { _ = lk.Close() })

	closed, err := ctx.NewLinker(JITOptions{})
	if err != nil {
		t.Fatalf("NewLinker closed: %v", err)
	}
	if err := closed.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil context", func() error { var c *Context; _, e := c.NewLinker(JITOptions{}); return e }, ErrNilContext},
		{"bad log buffer negative", func() error { _, e := ctx.NewLinker(JITOptions{LogBufferBytes: -1}); return e }, ErrInvalidLength},
		{"bad log buffer oversized", func() error { _, e := ctx.NewLinker(JITOptions{LogBufferBytes: maxJITLogBytes + 1}); return e }, ErrInvalidLength},
		{"nil linker AddPTX", func() error { var l *Linker; return l.AddPTX("k", []byte("x")) }, ErrNilLinker},
		{"nil linker AddCubin", func() error { var l *Linker; return l.AddCubin("k", []byte("x")) }, ErrNilLinker},
		{"nil linker Complete", func() error { var l *Linker; _, e := l.Complete(); return e }, ErrNilLinker},
		{"nil linker Close", func() error { var l *Linker; return l.Close() }, ErrNilLinker},
		{"max registers overflow", func() error {
			_, e := ctx.NewLinker(JITOptions{MaxRegisters: math.MaxUint32 + 1})
			return e
		}, ErrInvalidValue},
		{"empty ptx", func() error { return lk.AddPTX("k", nil) }, ErrEmptyImage},
		{"empty cubin", func() error { return lk.AddCubin("k", []byte{}) }, ErrEmptyImage},
		{"closed AddPTX", func() error { return closed.AddPTX("k", []byte("x")) }, ErrLinkerClosed},
		{"closed AddCubin", func() error { return closed.AddCubin("k", []byte("x")) }, ErrLinkerClosed},
		{"closed Complete", func() error { _, e := closed.Complete(); return e }, ErrLinkerClosed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestLinkerCloseRacesAdd guards the opMu contract: adds racing a Close must
// either complete before the destroy or fail with ErrLinkerClosed, never reach
// the driver with a destroyed link state.
func TestLinkerCloseRacesAdd(t *testing.T) {
	f := &linkFake{cubin: []byte{1}}
	ctx := newTestContext(t, linkDriver(f))
	lk, err := ctx.NewLinker(JITOptions{})
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := lk.AddPTX("k", []byte("x")); err != nil && !errors.Is(err, ErrLinkerClosed) {
				t.Errorf("AddPTX: %v", err)
			}
		}()
	}
	if err := lk.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	wg.Wait()
	if f.destroyCalls.Load() != 1 {
		t.Errorf("destroy calls = %d, want 1", f.destroyCalls.Load())
	}
	if f.addAfterDestroy.Load() {
		t.Error("an add reached the driver after the link state was destroyed")
	}
}

// TestLinkerCloseRetry guards the failed-destroy path: the linker stays open and
// usable, and a later Close retries the destroy.
func TestLinkerCloseRetry(t *testing.T) {
	f := &linkFake{cubin: []byte{1}}
	ctx := newTestContext(t, linkDriver(f))
	lk, err := ctx.NewLinker(JITOptions{})
	if err != nil {
		t.Fatalf("NewLinker: %v", err)
	}
	f.failDestroys.Store(1)
	if err := lk.Close(); err == nil {
		t.Fatal("expected the first Close to fail")
	}
	if err := lk.AddPTX("k", []byte("x")); err != nil {
		t.Errorf("AddPTX after failed Close: %v", err)
	}
	if err := lk.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if f.destroyCalls.Load() != 2 {
		t.Errorf("destroy calls = %d, want 2", f.destroyCalls.Load())
	}
}

// TestLinkerUnavailable guards the all-or-nothing feature group: a driver
// missing every cuLink symbol, or only some of them, must surface
// ErrSymbolUnavailable from NewLinker instead of creating an undestroyable state.
func TestLinkerUnavailable(t *testing.T) {
	ctx := newTestContext(t, linkBaseDriver())
	if _, err := ctx.NewLinker(JITOptions{}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("NewLinker = %v, want ErrSymbolUnavailable", err)
	}

	d := linkBaseDriver()
	d.CuLinkCreate = func(_ uint32, _ *int32, _ *uintptr, state *cudasys.CUlinkState) cudasys.CUresult {
		*state = 0x11AA
		return cudasys.CUDA_SUCCESS
	}
	partial := newTestContext(t, d)
	if _, err := partial.NewLinker(JITOptions{}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("NewLinker with partial group = %v, want ErrSymbolUnavailable", err)
	}
}

func TestLinkerLogNilReceiver(t *testing.T) {
	var l *Linker
	if got := l.Log(); got.Info != "" || got.Error != "" {
		t.Errorf("Log = %+v, want zero JITLog", got)
	}
}

func TestLinkerLogTrimsBuffers(t *testing.T) {
	lk := &Linker{
		infoBuf: []byte("info log\x00trailing garbage"),
		errBuf:  []byte("error log\x00\x00"),
	}
	got := lk.Log()
	if got.Info != "info log" {
		t.Errorf("Log.Info = %q, want %q", got.Info, "info log")
	}
	if got.Error != "error log" {
		t.Errorf("Log.Error = %q, want %q", got.Error, "error log")
	}
}
