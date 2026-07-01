package cuda

import (
	"context"
	"sync"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// memOpKind selects which driver call a memOp performs.
type memOpKind uint8

const (
	opHtoD memOpKind = iota
	opDtoH
	opDtoD
	opHtoDAsync
	opDtoHAsync
	opDtoDAsync
	opMemset8
	opMemset16
	opMemset32
	opMemset8Async
	opMemset16Async
	opMemset32Async
)

// memOp is a pooled, reusable memory operation. Submitting one to the executor
// instead of a closure keeps the copy and memset hot paths allocation-free. One
// struct covers every copy and memset variant; Run reads only the fields its
// kind needs, so unused fields left over from a previous use are harmless.
type memOp struct {
	driver *cudasys.Driver
	dst    cudasys.CUdeviceptr
	src    cudasys.CUdeviceptr
	host   *byte
	n      uint64
	val    uint32
	stream cudasys.CUstream
	kind   memOpKind
}

func (o *memOp) Run() error {
	switch o.kind {
	case opHtoD:
		return cudaresult.MemcpyHtoD(o.driver, o.dst, o.host, o.n)
	case opDtoH:
		return cudaresult.MemcpyDtoH(o.driver, o.host, o.src, o.n)
	case opDtoD:
		return cudaresult.MemcpyDtoD(o.driver, o.dst, o.src, o.n)
	case opHtoDAsync:
		return cudaresult.MemcpyHtoDAsync(o.driver, o.dst, o.host, o.n, o.stream)
	case opDtoHAsync:
		return cudaresult.MemcpyDtoHAsync(o.driver, o.host, o.src, o.n, o.stream)
	case opDtoDAsync:
		return cudaresult.MemcpyDtoDAsync(o.driver, o.dst, o.src, o.n, o.stream)
	case opMemset8:
		return cudaresult.MemsetD8(o.driver, o.dst, uint8(o.val), o.n)
	case opMemset16:
		return cudaresult.MemsetD16(o.driver, o.dst, uint16(o.val), o.n)
	case opMemset32:
		return cudaresult.MemsetD32(o.driver, o.dst, o.val, o.n)
	case opMemset8Async:
		return cudaresult.MemsetD8Async(o.driver, o.dst, uint8(o.val), o.n, o.stream)
	case opMemset16Async:
		return cudaresult.MemsetD16Async(o.driver, o.dst, uint16(o.val), o.n, o.stream)
	default: // opMemset32Async
		return cudaresult.MemsetD32Async(o.driver, o.dst, o.val, o.n, o.stream)
	}
}

var memOpPool = sync.Pool{New: func() any { return new(memOp) }}

// run submits o to the context executor (wait semantics) and returns it to the
// pool. doJob only returns after the executor is done with o, so recycling here
// is safe.
func (c *Context) run(ctx context.Context, o *memOp) error {
	o.driver = c.driver
	err := c.doJob(ctx, o)
	memOpPool.Put(o)
	return err
}

func (c *Context) memcpyHtoD(ctx context.Context, dst cudasys.CUdeviceptr, host *byte, n uint64) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.dst, o.host, o.n = opHtoD, dst, host, n
	return c.run(ctx, o)
}

func (c *Context) memcpyDtoH(ctx context.Context, host *byte, src cudasys.CUdeviceptr, n uint64) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.host, o.src, o.n = opDtoH, host, src, n
	return c.run(ctx, o)
}

func (c *Context) memcpyDtoD(ctx context.Context, dst, src cudasys.CUdeviceptr, n uint64) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.dst, o.src, o.n = opDtoD, dst, src, n
	return c.run(ctx, o)
}

func (c *Context) memcpyHtoDAsync(ctx context.Context, dst cudasys.CUdeviceptr, host *byte, n uint64, stream cudasys.CUstream) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.dst, o.host, o.n, o.stream = opHtoDAsync, dst, host, n, stream
	return c.run(ctx, o)
}

func (c *Context) memcpyDtoHAsync(ctx context.Context, host *byte, src cudasys.CUdeviceptr, n uint64, stream cudasys.CUstream) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.host, o.src, o.n, o.stream = opDtoHAsync, host, src, n, stream
	return c.run(ctx, o)
}

func (c *Context) memcpyDtoDAsync(ctx context.Context, dst, src cudasys.CUdeviceptr, n uint64, stream cudasys.CUstream) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.dst, o.src, o.n, o.stream = opDtoDAsync, dst, src, n, stream
	return c.run(ctx, o)
}

// memsetKind maps an element width in bytes to the matching memset kind. base is
// opMemset8 for the synchronous set and opMemset8Async for the stream variant.
func memsetKind(base memOpKind, size uintptr) memOpKind {
	switch size {
	case 1:
		return base
	case 2:
		return base + 1
	default:
		return base + 2
	}
}

func (c *Context) memset(ctx context.Context, dst cudasys.CUdeviceptr, val uint32, n uint64, size uintptr) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.dst, o.val, o.n = memsetKind(opMemset8, size), dst, val, n
	return c.run(ctx, o)
}

func (c *Context) memsetAsync(ctx context.Context, dst cudasys.CUdeviceptr, val uint32, n uint64, size uintptr, stream cudasys.CUstream) error {
	o := memOpPool.Get().(*memOp)
	o.kind, o.dst, o.val, o.n, o.stream = memsetKind(opMemset8Async, size), dst, val, n, stream
	return c.run(ctx, o)
}

// launchOp is a pooled kernel-launch operation, submitted to the executor
// without a per-launch closure.
type launchOp struct {
	driver *cudasys.Driver
	params *unsafe.Pointer
	cfg    LaunchConfig
	fn     cudasys.CUfunction
	stream cudasys.CUstream
	coop   bool
}

func (o *launchOp) Run() error {
	launch := cudaresult.LaunchKernel
	if o.coop {
		launch = cudaresult.LaunchCooperativeKernel
	}
	return launch(
		o.driver, o.fn,
		o.cfg.GridX, o.cfg.GridY, o.cfg.GridZ,
		o.cfg.BlockX, o.cfg.BlockY, o.cfg.BlockZ,
		o.cfg.SharedMemBytes, o.stream, o.params,
	)
}

var launchOpPool = sync.Pool{New: func() any { return new(launchOp) }}

func (c *Context) launchKernel(ctx context.Context, fn cudasys.CUfunction, cfg LaunchConfig, stream cudasys.CUstream, params *unsafe.Pointer, cooperative bool) error {
	o := launchOpPool.Get().(*launchOp)
	o.driver, o.fn, o.cfg, o.stream, o.params, o.coop = c.driver, fn, cfg, stream, params, cooperative
	err := c.doJob(ctx, o)
	launchOpPool.Put(o)
	return err
}

// graphLaunchOp is a pooled executable-graph launch.
type graphLaunchOp struct {
	driver *cudasys.Driver
	exec   cudasys.CUgraphExec
	stream cudasys.CUstream
}

func (o *graphLaunchOp) Run() error {
	return cudaresult.GraphLaunch(o.driver, o.exec, o.stream)
}

var graphLaunchOpPool = sync.Pool{New: func() any { return new(graphLaunchOp) }}

func (c *Context) graphLaunch(ctx context.Context, exec cudasys.CUgraphExec, stream cudasys.CUstream) error {
	o := graphLaunchOpPool.Get().(*graphLaunchOp)
	o.driver, o.exec, o.stream = c.driver, exec, stream
	err := c.doJob(ctx, o)
	graphLaunchOpPool.Put(o)
	return err
}
