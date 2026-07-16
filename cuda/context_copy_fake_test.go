package cuda

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/eitamring/gocudrv/cudasys"
)

func hasCopyExecutor(c *Context) bool {
	c.copyMu.Lock()
	defer c.copyMu.Unlock()
	return c.copyExec != nil
}

func copyLaneDriver(c *waitLaneCalls) *cudasys.Driver {
	driver := waitLaneDriver(c)
	driver.CuMemcpyHtoD = func(cudasys.CUdeviceptr, *byte, uint64) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	driver.CuMemcpyDtoH = func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	driver.CuMemcpyDtoD = func(cudasys.CUdeviceptr, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	driver.CuMemcpy2D = func(*cudasys.Memcpy2D) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	driver.CuMemcpy3D = func(*cudasys.Memcpy3D) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	driver.CuMemcpyPeer = func(cudasys.CUdeviceptr, cudasys.CUcontext, cudasys.CUdeviceptr, cudasys.CUcontext, uint64) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	return driver
}

func TestCopyExecutorIsLazy(t *testing.T) {
	var calls waitLaneCalls
	ctx := newTestContext(t, copyLaneDriver(&calls))
	if hasCopyExecutor(ctx) {
		t.Fatal("copy executor exists before the first copy")
	}
	if _, _, err := ctx.MemInfo(); err != nil {
		t.Fatalf("MemInfo: %v", err)
	}
	if hasCopyExecutor(ctx) {
		t.Fatal("ordinary command created the copy executor")
	}
	var dst byte
	if err := ctx.memcpyDtoH(context.Background(), &dst, 0xDEAD, 1); err != nil {
		t.Fatalf("copy: %v", err)
	}
	if !hasCopyExecutor(ctx) {
		t.Fatal("copy did not create the copy executor")
	}
}

func TestPreCanceledCopyDoesNotCreateExecutor(t *testing.T) {
	var calls waitLaneCalls
	driver := copyLaneDriver(&calls)
	copyCalls := atomic.Int32{}
	driver.CuMemcpyDtoH = func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		copyCalls.Add(1)
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	waitCtx, cancel := context.WithCancel(context.Background())
	cancel()
	var dst byte
	if err := ctx.memcpyDtoH(waitCtx, &dst, 0xDEAD, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("copy = %v, want context.Canceled", err)
	}
	if hasCopyExecutor(ctx) {
		t.Fatal("pre-canceled copy created the copy executor")
	}
	if copyCalls.Load() != 0 {
		t.Fatalf("copy calls = %d, want 0", copyCalls.Load())
	}
}

func TestAsyncCopyStaysOnCommandExecutor(t *testing.T) {
	var calls waitLaneCalls
	driver := copyLaneDriver(&calls)
	driver.CuMemcpyDtoHAsync = func(*byte, cudasys.CUdeviceptr, uint64, cudasys.CUstream) cudasys.CUresult {
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	var dst byte
	if err := ctx.memcpyDtoHAsync(context.Background(), &dst, 0xDEAD, 1, 0x5151); err != nil {
		t.Fatalf("async copy: %v", err)
	}
	if hasCopyExecutor(ctx) {
		t.Fatal("async copy created the blocking copy executor")
	}
}

func TestBlockingCopyLeavesCommandExecutorFree(t *testing.T) {
	var calls waitLaneCalls
	driver := copyLaneDriver(&calls)
	copyEntered := make(chan struct{})
	releaseCopy := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(releaseCopy) }) }
	defer unblock()
	driver.CuMemcpyDtoH = func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		close(copyEntered)
		<-releaseCopy
		return cudasys.CUDA_SUCCESS
	}

	ctx := newTestContext(t, driver)
	stream := &Stream{ctx: ctx, raw: 0x5151}
	var dst byte
	copyDone := make(chan error, 1)
	go func() {
		copyDone <- ctx.memcpyDtoH(context.Background(), &dst, 0xDEAD, 1)
	}()
	waitSignal(t, copyEntered, "blocking copy")

	commandDone := make(chan error, 1)
	go func() { commandDone <- stream.Query() }()
	select {
	case err := <-commandDone:
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
	case <-time.After(250 * time.Millisecond):
		unblock()
		t.Fatal("Stream.Query blocked behind a synchronous copy")
	}

	unblock()
	if err := waitError(t, copyDone, "blocking copy completion"); err != nil {
		t.Fatalf("copy: %v", err)
	}
}

func TestSynchronousCopyKindsUseCopyExecutor(t *testing.T) {
	var host byte
	tests := []struct {
		name string
		copy func(*Context) error
	}{
		{"host to device", func(ctx *Context) error {
			return ctx.memcpyHtoD(context.Background(), 0xCAFE, &host, 1)
		}},
		{"device to host", func(ctx *Context) error {
			return ctx.memcpyDtoH(context.Background(), &host, 0xCAFE, 1)
		}},
		{"device to device", func(ctx *Context) error {
			return ctx.memcpyDtoD(context.Background(), 0xCAFE, 0xDEAD, 1)
		}},
		{"2d", func(ctx *Context) error {
			return ctx.memcpy2D(context.Background(), &cudasys.Memcpy2D{})
		}},
		{"3d", func(ctx *Context) error {
			return ctx.memcpy3D(context.Background(), &cudasys.Memcpy3D{})
		}},
		{"peer", func(ctx *Context) error {
			src := &Buffer[byte]{ctx: ctx, ptr: 0xCAFE, length: 1, bytes: 1}
			dst := &Buffer[byte]{ctx: ctx, ptr: 0xDEAD, length: 1, bytes: 1}
			return src.CopyToPeer(context.Background(), dst)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var calls waitLaneCalls
			ctx := newTestContext(t, copyLaneDriver(&calls))
			commandEntered := make(chan struct{})
			releaseCommand := make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(releaseCommand) }) }
			defer unblock()
			commandDone := make(chan error, 1)
			go func() {
				commandDone <- ctx.exec.Do(func() error {
					close(commandEntered)
					<-releaseCommand
					return nil
				})
			}()
			waitSignal(t, commandEntered, "blocked command")

			copyDone := make(chan error, 1)
			go func() { copyDone <- tc.copy(ctx) }()
			if err := waitError(t, copyDone, "copy"); err != nil {
				t.Fatalf("copy: %v", err)
			}
			unblock()
			if err := waitError(t, commandDone, "command completion"); err != nil {
				t.Fatalf("command: %v", err)
			}
		})
	}
}

func TestCopyExecutorBindFailureCanRetry(t *testing.T) {
	var calls waitLaneCalls
	driver := copyLaneDriver(&calls)
	var rawAttempts atomic.Int32
	driver.CuCtxSetCurrent = func(ctx cudasys.CUcontext) cudasys.CUresult {
		if ctx == 0 {
			calls.zeroBinds.Add(1)
			return cudasys.CUDA_SUCCESS
		}
		attempt := rawAttempts.Add(1)
		calls.rawBinds.Add(1)
		if attempt == 2 {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	var dst byte

	if err := ctx.memcpyDtoH(context.Background(), &dst, 0xDEAD, 1); !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("first copy = %v, want ErrInvalidContext", err)
	}
	if hasCopyExecutor(ctx) {
		t.Fatal("failed bind stored a copy executor")
	}
	if err := ctx.memcpyDtoH(context.Background(), &dst, 0xDEAD, 1); err != nil {
		t.Fatalf("retry copy: %v", err)
	}
	if !hasCopyExecutor(ctx) {
		t.Fatal("successful retry did not store a copy executor")
	}
	if rawAttempts.Load() != 3 {
		t.Fatalf("raw bind attempts = %d, want 3", rawAttempts.Load())
	}
}

func TestContextCloseWaitsForBlockingCopy(t *testing.T) {
	var calls waitLaneCalls
	driver := copyLaneDriver(&calls)
	copyEntered := make(chan struct{})
	releaseCopy := make(chan struct{})
	releaseCalled := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(releaseCopy) }) }
	defer unblock()
	driver.CuMemcpyDtoH = func(*byte, cudasys.CUdeviceptr, uint64) cudasys.CUresult {
		close(copyEntered)
		<-releaseCopy
		return cudasys.CUDA_SUCCESS
	}
	driver.CuDevicePrimaryCtxRelease = func(cudasys.CUdevice) cudasys.CUresult {
		calls.releases.Add(1)
		close(releaseCalled)
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	var dst byte
	copyDone := make(chan error, 1)
	go func() {
		copyDone <- ctx.memcpyDtoH(context.Background(), &dst, 0xDEAD, 1)
	}()
	waitSignal(t, copyEntered, "blocking copy")

	closeDone := make(chan error, 1)
	go func() { closeDone <- ctx.Close() }()
	assertPending(t, closeDone, "Context.Close")
	select {
	case <-releaseCalled:
		t.Fatal("primary context released before the copy finished")
	default:
	}
	unblock()
	if err := waitError(t, copyDone, "copy completion"); err != nil {
		t.Fatalf("copy: %v", err)
	}
	if err := waitError(t, closeDone, "Context.Close completion"); err != nil {
		t.Fatalf("Close: %v", err)
	}
	waitSignal(t, releaseCalled, "primary context release")
}

func TestCopyExecutorUnbindFailureStillReleases(t *testing.T) {
	var calls waitLaneCalls
	driver := copyLaneDriver(&calls)
	driver.CuCtxSetCurrent = func(ctx cudasys.CUcontext) cudasys.CUresult {
		if ctx != 0 {
			calls.rawBinds.Add(1)
			return cudasys.CUDA_SUCCESS
		}
		attempt := calls.zeroBinds.Add(1)
		if attempt == 1 {
			return cudasys.CUDA_ERROR_INVALID_CONTEXT
		}
		return cudasys.CUDA_SUCCESS
	}
	ctx := newTestContext(t, driver)
	var dst byte
	if err := ctx.memcpyDtoH(context.Background(), &dst, 0xDEAD, 1); err != nil {
		t.Fatalf("copy: %v", err)
	}
	if err := ctx.Close(); !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("Close = %v, want ErrInvalidContext", err)
	}
	if calls.releases.Load() != 1 {
		t.Fatalf("release calls = %d, want 1", calls.releases.Load())
	}
	if !ctx.closed.Load() {
		t.Fatal("context left open after successful release")
	}
}
