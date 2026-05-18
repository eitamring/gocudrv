package cuda

import (
	"context"
	"math"
	"sync"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/argpack"
)

// LaunchConfig describes the grid/block shape for a kernel launch.
type LaunchConfig struct {
	GridX, GridY, GridZ    uint32
	BlockX, BlockY, BlockZ uint32
	SharedMemBytes         uint32
}

// LaunchConfig1D returns a one-dimensional launch shape covering n elements.
// Invalid inputs return an empty config, which Launch rejects with
// ErrInvalidLaunchConfig.
func LaunchConfig1D(n, blockSize int) LaunchConfig {
	if n <= 0 || blockSize <= 0 || uint64(blockSize) > math.MaxUint32 {
		return LaunchConfig{}
	}
	grid := (uint64(n) + uint64(blockSize) - 1) / uint64(blockSize)
	if grid == 0 || grid > math.MaxUint32 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX:  uint32(grid),
		GridY:  1,
		GridZ:  1,
		BlockX: uint32(blockSize),
		BlockY: 1,
		BlockZ: 1,
	}
}

func (cfg LaunchConfig) valid() bool {
	return cfg.GridX > 0 && cfg.GridY > 0 && cfg.GridZ > 0 &&
		cfg.BlockX > 0 && cfg.BlockY > 0 && cfg.BlockZ > 0
}

type kernelArg interface {
	appendKernelArg(*kernelArgBuilder) error
}

// KernelArg is a value accepted by Function.Launch.
type KernelArg interface {
	kernelArg
}

type kernelArgBuilder struct {
	ctx           *Context
	packed        argpack.Builder
	inlineLocks   [16]*sync.RWMutex
	lockCount     int
	overflowLocks []*sync.RWMutex
}

func (b *kernelArgBuilder) addDevicePtr(ptr cudasys.CUdeviceptr) {
	argpack.Add(&b.packed, ptr)
}

func (b *kernelArgBuilder) addLock(mu *sync.RWMutex) {
	if b.lockCount < len(b.inlineLocks) {
		b.inlineLocks[b.lockCount] = mu
		b.lockCount++
		return
	}
	b.overflowLocks = append(b.overflowLocks, mu)
	b.lockCount++
}

func (b *kernelArgBuilder) release() {
	for i := b.lockCount - 1; i >= 0; i-- {
		if i < len(b.inlineLocks) {
			b.inlineLocks[i].RUnlock()
			continue
		}
		b.overflowLocks[i-len(b.inlineLocks)].RUnlock()
	}
}

type bufferKernelArg[T Supported] struct {
	buffer *Buffer[T]
}

// Arg passes a device buffer pointer to a kernel.
func Arg[T Supported](b *Buffer[T]) KernelArg {
	return bufferKernelArg[T]{buffer: b}
}

func (a bufferKernelArg[T]) appendKernelArg(b *kernelArgBuilder) error {
	if a.buffer == nil {
		return ErrNilBuffer
	}
	a.buffer.opMu.RLock()
	if a.buffer.closed {
		a.buffer.opMu.RUnlock()
		return ErrBufferClosed
	}
	if a.buffer.ctx != b.ctx {
		a.buffer.opMu.RUnlock()
		return ErrContextMismatch
	}
	b.addLock(&a.buffer.opMu)
	b.addDevicePtr(a.buffer.ptr)
	return nil
}

type valueKernelArg[T Supported] struct {
	value T
}

// ArgValue passes a scalar value to a kernel.
func ArgValue[T Supported](v T) KernelArg {
	return valueKernelArg[T]{value: v}
}

func (a valueKernelArg[T]) appendKernelArg(b *kernelArgBuilder) error {
	argpack.Add(&b.packed, a.value)
	return nil
}

// Launch enqueues the function on the context's legacy default stream.
// Cancellation can prevent submission, but once submitted Launch waits until
// cuLaunchKernel returns so temporary argument storage remains valid. The
// kernel itself may still be running after Launch returns; call
// Context.Synchronize before reading outputs or closing resources used by the
// launch.
func (f *Function) Launch(ctx context.Context, cfg LaunchConfig, args ...KernelArg) error {
	return f.launch(ctx, defaultStream, nil, cfg, args...)
}

// LaunchOn enqueues the function on stream. The stream must belong to the same
// Context as the Function. Cancellation and resource-lifetime rules match
// Launch; use stream.Synchronize to wait for work submitted to that stream.
func (f *Function) LaunchOn(ctx context.Context, stream *Stream, cfg LaunchConfig, args ...KernelArg) error {
	if f == nil {
		return ErrNilFunction
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	return f.launch(ctx, stream.raw, stream.ctx, cfg, args...)
}

func (f *Function) launch(ctx context.Context, rawStream cudasys.CUstream, streamCtx *Context, cfg LaunchConfig, args ...KernelArg) error {
	if f == nil {
		return ErrNilFunction
	}
	if !cfg.valid() {
		return ErrInvalidLaunchConfig
	}
	if f.module == nil {
		return ErrNilModule
	}
	if streamCtx != nil && streamCtx != f.module.ctx {
		return ErrContextMismatch
	}

	f.module.opMu.RLock()
	defer f.module.opMu.RUnlock()
	if f.module.closed {
		return ErrModuleClosed
	}

	builder := kernelArgBuilder{ctx: f.module.ctx}
	defer builder.release()
	for _, arg := range args {
		if arg == nil {
			return ErrNilKernelArg
		}
		if err := arg.appendKernelArg(&builder); err != nil {
			return err
		}
	}

	err := f.module.ctx.doWait(ctx, func() error {
		return cudaresult.LaunchKernel(
			f.module.ctx.driver,
			f.raw,
			cfg.GridX, cfg.GridY, cfg.GridZ,
			cfg.BlockX, cfg.BlockY, cfg.BlockZ,
			cfg.SharedMemBytes,
			rawStream,
			builder.packed.Params(),
		)
	})
	builder.packed.KeepAlive()
	return err
}

const defaultStream cudasys.CUstream = 0
