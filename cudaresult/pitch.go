package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// MemAllocPitch allocates a 2D region of widthInBytes by height rows, letting
// the driver choose a row pitch aligned for coalesced access. It returns the
// device pointer and the pitch in bytes. elementSizeBytes must be 4, 8, or 16.
// Bound best-effort, so it returns ErrSymbolUnavailable on a driver that lacks
// the symbol.
func MemAllocPitch(d *cudasys.Driver, widthInBytes, height uint64, elementSizeBytes uint32) (cudasys.CUdeviceptr, uint64, error) {
	if d == nil {
		return 0, 0, ErrNotInitialized
	}
	if d.CuMemAllocPitch == nil {
		return 0, 0, ErrSymbolUnavailable
	}
	var ptr cudasys.CUdeviceptr
	var pitch uint64
	if err := check("cuMemAllocPitch_v2", d.CuMemAllocPitch(&ptr, &pitch, widthInBytes, height, elementSizeBytes)); err != nil {
		return 0, 0, err
	}
	return ptr, pitch, nil
}

// Memcpy2D performs the 2D copy described by desc.
func Memcpy2D(d *cudasys.Driver, desc *cudasys.Memcpy2D) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemcpy2D == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemcpy2D_v2", d.CuMemcpy2D(desc))
}

// Memcpy2DAsync performs the 2D copy described by desc, enqueued on stream.
func Memcpy2DAsync(d *cudasys.Driver, desc *cudasys.Memcpy2D, stream cudasys.CUstream) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemcpy2DAsync == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemcpy2DAsync_v2", d.CuMemcpy2DAsync(desc, stream))
}

// Memcpy3D performs the 3D copy described by desc.
func Memcpy3D(d *cudasys.Driver, desc *cudasys.Memcpy3D) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemcpy3D == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemcpy3D_v2", d.CuMemcpy3D(desc))
}

// Memcpy3DAsync performs the 3D copy described by desc, enqueued on stream.
func Memcpy3DAsync(d *cudasys.Driver, desc *cudasys.Memcpy3D, stream cudasys.CUstream) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemcpy3DAsync == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemcpy3DAsync_v2", d.CuMemcpy3DAsync(desc, stream))
}
