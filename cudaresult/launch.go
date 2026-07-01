package cudaresult

import (
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// LaunchKernel launches fn with the supplied execution configuration and
// parameter pointer array on stream.
func LaunchKernel(
	d *cudasys.Driver,
	fn cudasys.CUfunction,
	gridX, gridY, gridZ uint32,
	blockX, blockY, blockZ uint32,
	sharedMemBytes uint32,
	stream cudasys.CUstream,
	kernelParams *unsafe.Pointer,
) error {
	if d == nil || d.CuLaunchKernel == nil {
		return ErrNotInitialized
	}
	return check("cuLaunchKernel", d.CuLaunchKernel(
		fn,
		gridX, gridY, gridZ,
		blockX, blockY, blockZ,
		sharedMemBytes,
		stream,
		kernelParams,
		nil,
	))
}

// LaunchCooperativeKernel launches fn as a cooperative kernel, whose grid can
// synchronize as a whole, with the supplied execution configuration and
// parameter pointer array on stream. It returns ErrSymbolUnavailable on a
// driver that does not export cuLaunchCooperativeKernel.
func LaunchCooperativeKernel(
	d *cudasys.Driver,
	fn cudasys.CUfunction,
	gridX, gridY, gridZ uint32,
	blockX, blockY, blockZ uint32,
	sharedMemBytes uint32,
	stream cudasys.CUstream,
	kernelParams *unsafe.Pointer,
) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuLaunchCooperativeKernel == nil {
		return ErrSymbolUnavailable
	}
	return check("cuLaunchCooperativeKernel", d.CuLaunchCooperativeKernel(
		fn,
		gridX, gridY, gridZ,
		blockX, blockY, blockZ,
		sharedMemBytes,
		stream,
		kernelParams,
	))
}
