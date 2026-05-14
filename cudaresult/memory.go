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
