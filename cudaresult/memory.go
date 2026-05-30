package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// MemAlloc allocates bytes of device memory and returns the opaque pointer.
func MemAlloc(d *cudasys.Driver, bytes uint64) (cudasys.CUdeviceptr, error) {
	if d == nil || d.CuMemAlloc == nil {
		return 0, ErrNotInitialized
	}
	var ptr cudasys.CUdeviceptr
	if err := check("cuMemAlloc_v2", d.CuMemAlloc(&ptr, bytes)); err != nil {
		return 0, err
	}
	return ptr, nil
}

// MemFree releases device memory previously returned by MemAlloc.
func MemFree(d *cudasys.Driver, ptr cudasys.CUdeviceptr) error {
	if d == nil || d.CuMemFree == nil {
		return ErrNotInitialized
	}
	return check("cuMemFree_v2", d.CuMemFree(ptr))
}

// MemGetInfo returns the free and total device memory in bytes for the current
// context's device.
func MemGetInfo(d *cudasys.Driver) (free, total uint64, err error) {
	if d == nil || d.CuMemGetInfo == nil {
		return 0, 0, ErrNotInitialized
	}
	if err := check("cuMemGetInfo_v2", d.CuMemGetInfo(&free, &total)); err != nil {
		return 0, 0, err
	}
	return free, total, nil
}

// MemcpyHtoD copies bytes from a host pointer to device memory and blocks
// until the copy finishes.
func MemcpyHtoD(d *cudasys.Driver, dst cudasys.CUdeviceptr, src *byte, bytes uint64) error {
	if d == nil || d.CuMemcpyHtoD == nil {
		return ErrNotInitialized
	}
	return check("cuMemcpyHtoD_v2", d.CuMemcpyHtoD(dst, src, bytes))
}

// MemcpyDtoH copies bytes from device memory to a host pointer and blocks
// until the copy finishes.
func MemcpyDtoH(d *cudasys.Driver, dst *byte, src cudasys.CUdeviceptr, bytes uint64) error {
	if d == nil || d.CuMemcpyDtoH == nil {
		return ErrNotInitialized
	}
	return check("cuMemcpyDtoH_v2", d.CuMemcpyDtoH(dst, src, bytes))
}

// MemcpyHtoDAsync enqueues a host-to-device copy on stream and returns after
// the driver accepts the work.
func MemcpyHtoDAsync(d *cudasys.Driver, dst cudasys.CUdeviceptr, src *byte, bytes uint64, stream cudasys.CUstream) error {
	if d == nil || d.CuMemcpyHtoDAsync == nil {
		return ErrNotInitialized
	}
	return check("cuMemcpyHtoDAsync_v2", d.CuMemcpyHtoDAsync(dst, src, bytes, stream))
}

// MemcpyDtoHAsync enqueues a device-to-host copy on stream and returns after
// the driver accepts the work.
func MemcpyDtoHAsync(d *cudasys.Driver, dst *byte, src cudasys.CUdeviceptr, bytes uint64, stream cudasys.CUstream) error {
	if d == nil || d.CuMemcpyDtoHAsync == nil {
		return ErrNotInitialized
	}
	return check("cuMemcpyDtoHAsync_v2", d.CuMemcpyDtoHAsync(dst, src, bytes, stream))
}

// MemcpyDtoD copies bytes between two device pointers and blocks until the copy
// finishes.
func MemcpyDtoD(d *cudasys.Driver, dst, src cudasys.CUdeviceptr, bytes uint64) error {
	if d == nil || d.CuMemcpyDtoD == nil {
		return ErrNotInitialized
	}
	return check("cuMemcpyDtoD_v2", d.CuMemcpyDtoD(dst, src, bytes))
}

// MemcpyDtoDAsync enqueues a device-to-device copy on stream and returns after
// the driver accepts the work.
func MemcpyDtoDAsync(d *cudasys.Driver, dst, src cudasys.CUdeviceptr, bytes uint64, stream cudasys.CUstream) error {
	if d == nil || d.CuMemcpyDtoDAsync == nil {
		return ErrNotInitialized
	}
	return check("cuMemcpyDtoDAsync_v2", d.CuMemcpyDtoDAsync(dst, src, bytes, stream))
}

// MemsetD8 sets count bytes at dst to value and blocks until it finishes.
func MemsetD8(d *cudasys.Driver, dst cudasys.CUdeviceptr, value uint8, count uint64) error {
	if d == nil || d.CuMemsetD8 == nil {
		return ErrNotInitialized
	}
	return check("cuMemsetD8_v2", d.CuMemsetD8(dst, value, count))
}

// MemsetD32 sets count 32-bit words at dst to value and blocks until it
// finishes.
func MemsetD32(d *cudasys.Driver, dst cudasys.CUdeviceptr, value uint32, count uint64) error {
	if d == nil || d.CuMemsetD32 == nil {
		return ErrNotInitialized
	}
	return check("cuMemsetD32_v2", d.CuMemsetD32(dst, value, count))
}

// MemsetD8Async enqueues a byte memset on stream and returns after the driver
// accepts the work.
func MemsetD8Async(d *cudasys.Driver, dst cudasys.CUdeviceptr, value uint8, count uint64, stream cudasys.CUstream) error {
	if d == nil || d.CuMemsetD8Async == nil {
		return ErrNotInitialized
	}
	return check("cuMemsetD8Async", d.CuMemsetD8Async(dst, value, count, stream))
}

// MemsetD32Async enqueues a 32-bit-word memset on stream and returns after the
// driver accepts the work.
func MemsetD32Async(d *cudasys.Driver, dst cudasys.CUdeviceptr, value uint32, count uint64, stream cudasys.CUstream) error {
	if d == nil || d.CuMemsetD32Async == nil {
		return ErrNotInitialized
	}
	return check("cuMemsetD32Async", d.CuMemsetD32Async(dst, value, count, stream))
}

// MemAllocHost allocates bytes of page-locked host memory and returns the
// host pointer. The pointer is suitable for direct DMA by the GPU.
func MemAllocHost(d *cudasys.Driver, bytes uint64) (*byte, error) {
	if d == nil || d.CuMemAllocHost == nil {
		return nil, ErrNotInitialized
	}
	var p *byte
	if err := check("cuMemAllocHost_v2", d.CuMemAllocHost(&p, bytes)); err != nil {
		return nil, err
	}
	return p, nil
}

// MemFreeHost releases page-locked host memory previously returned by
// MemAllocHost.
func MemFreeHost(d *cudasys.Driver, p *byte) error {
	if d == nil || d.CuMemFreeHost == nil {
		return ErrNotInitialized
	}
	return check("cuMemFreeHost", d.CuMemFreeHost(p))
}
