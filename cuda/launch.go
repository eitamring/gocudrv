package cuda

import (
	"context"
	"math"
	"runtime"
	"sync"
	"unsafe"

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
	packed        *argpack.Builder
	inlineLocks   [16]*sync.RWMutex
	lockCount     int
	overflowLocks []*sync.RWMutex
	// snapshot packs a buffer's device pointer without retaining its lock, for
	// PackedArgs where the args outlive a single launch and the caller owns
	// buffer lifetime.
	snapshot bool
}

func (b *kernelArgBuilder) addDevicePtr(ptr cudasys.CUdeviceptr) {
	argpack.Add(b.packed, ptr)
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

// holdsLock reports whether mu is already held by this builder, so an argument
// repeated in one launch does not re-acquire it: a second RLock could deadlock
// behind a Close that is already waiting on the first.
func (b *kernelArgBuilder) holdsLock(mu *sync.RWMutex) bool {
	for i := 0; i < b.lockCount && i < len(b.inlineLocks); i++ {
		if b.inlineLocks[i] == mu {
			return true
		}
	}
	for _, held := range b.overflowLocks {
		if held == mu {
			return true
		}
	}
	return false
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
	held := b.holdsLock(&a.buffer.opMu)
	if !held {
		a.buffer.opMu.RLock()
	}
	if a.buffer.closed {
		if !held {
			a.buffer.opMu.RUnlock()
		}
		return ErrBufferClosed
	}
	if b.ctx != nil && a.buffer.ctx != b.ctx {
		if !held {
			a.buffer.opMu.RUnlock()
		}
		return ErrContextMismatch
	}
	ptr := a.buffer.ptr
	if b.snapshot {
		if !held {
			a.buffer.opMu.RUnlock()
		}
	} else if !held {
		b.addLock(&a.buffer.opMu)
	}
	b.addDevicePtr(ptr)
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
	argpack.Add(b.packed, a.value)
	return nil
}

// maxRawArgBytes bounds a single raw kernel argument; CUDA's parameter space is a few KiB.
const maxRawArgBytes = 4096

type devicePtrKernelArg struct {
	ptr cudasys.CUdeviceptr
}

// ArgDevicePtr passes a raw device pointer to a kernel for a caller that holds
// one directly. It takes no lock and tracks no buffer lifetime.
func ArgDevicePtr(ptr cudasys.CUdeviceptr) KernelArg {
	return devicePtrKernelArg{ptr: ptr}
}

func (a devicePtrKernelArg) appendKernelArg(b *kernelArgBuilder) error {
	b.addDevicePtr(a.ptr)
	return nil
}

type rawKernelArg struct {
	value unsafe.Pointer
	size  int
}

// ArgRaw passes size bytes at value as a kernel argument, for types Arg and
// ArgValue do not cover. value must be non-nil and size in (0, 4096].
func ArgRaw(value unsafe.Pointer, size int) KernelArg {
	return rawKernelArg{value: value, size: size}
}

func (a rawKernelArg) appendKernelArg(b *kernelArgBuilder) error {
	if a.value == nil {
		return ErrNilKernelArg
	}
	if a.size <= 0 || a.size > maxRawArgBytes {
		return ErrInvalidArgSize
	}
	argpack.AddBytes(b.packed, unsafe.Slice((*byte)(a.value), a.size))
	return nil
}

// Launch enqueues the function on the context's legacy default stream.
// Cancellation can prevent submission, but once submitted Launch waits until
// cuLaunchKernel returns so temporary argument storage remains valid. The
// kernel itself may still be running after Launch returns; call
// Context.Synchronize before reading outputs or closing resources used by the
// launch.
func (f *Function) Launch(ctx context.Context, cfg LaunchConfig, args ...KernelArg) error {
	return f.launch(ctx, defaultStream, nil, cfg, false, args...)
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
	return f.launch(ctx, stream.raw, stream.ctx, cfg, false, args...)
}

func (f *Function) launch(ctx context.Context, rawStream cudasys.CUstream, streamCtx *Context, cfg LaunchConfig, cooperative bool, args ...KernelArg) error {
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

	var pk argpack.Builder
	builder := kernelArgBuilder{ctx: f.module.ctx, packed: &pk}
	defer builder.release()
	for _, arg := range args {
		if arg == nil {
			return ErrNilKernelArg
		}
		if err := arg.appendKernelArg(&builder); err != nil {
			return err
		}
	}

	err := f.module.ctx.launchKernel(ctx, f.raw, cfg, rawStream, pk.Params(), cooperative)
	pk.KeepAlive()
	return err
}

// PackedArgs is a kernel argument list packed once for repeated low-overhead
// launches. It captures raw handles at pack time and takes no per-launch locks,
// so keep every referenced Buffer and Texture (and its array) open and
// unchanged while it is used. Build one with Pack and launch it with
// Function.LaunchPacked; do not copy it, pass the pointer Pack returns.
type PackedArgs struct {
	packed argpack.Builder
}

// Pack packs args into a reusable PackedArgs. It accepts the same arguments as
// Launch, but resolves each buffer or texture to a raw handle immediately
// rather than holding its lock, so the caller owns lifetime from here on. Pack
// does no context check; the resources must belong to the Context the kernel is
// launched on.
func Pack(args ...KernelArg) (*PackedArgs, error) {
	p := &PackedArgs{}
	b := kernelArgBuilder{packed: &p.packed, snapshot: true}
	for _, arg := range args {
		if arg == nil {
			return nil, ErrNilKernelArg
		}
		if err := arg.appendKernelArg(&b); err != nil {
			return nil, err
		}
	}
	return p, nil
}

// Len returns the number of packed arguments. It is 0 for a nil receiver.
func (p *PackedArgs) Len() int {
	if p == nil {
		return 0
	}
	return p.packed.Len()
}

// LaunchPacked launches f on the context's legacy default stream with
// pre-packed arguments and no per-launch argument allocation. The lifetime
// rules of PackedArgs apply: keep the referenced buffers open until the launch
// has been synchronized.
func (f *Function) LaunchPacked(ctx context.Context, cfg LaunchConfig, p *PackedArgs) error {
	return f.launchPacked(ctx, defaultStream, nil, cfg, p)
}

// LaunchPackedOn is LaunchPacked on a specific stream of the same Context.
func (f *Function) LaunchPackedOn(ctx context.Context, stream *Stream, cfg LaunchConfig, p *PackedArgs) error {
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
	return f.launchPacked(ctx, stream.raw, stream.ctx, cfg, p)
}

func (f *Function) launchPacked(ctx context.Context, rawStream cudasys.CUstream, streamCtx *Context, cfg LaunchConfig, p *PackedArgs) error {
	if f == nil {
		return ErrNilFunction
	}
	if p == nil {
		return ErrNilKernelArg
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
	err := f.module.ctx.launchKernel(ctx, f.raw, cfg, rawStream, p.packed.Params(), false)
	runtime.KeepAlive(p)
	return err
}

const defaultStream cudasys.CUstream = 0
