package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// MemGetAllocationGranularity returns the allocation granularity for prop.
// option selects the minimum or recommended granularity.
func MemGetAllocationGranularity(d *cudasys.Driver, prop *cudasys.CUmemAllocationProp, option uint32) (uint64, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuMemGetAllocationGranularity == nil {
		return 0, ErrSymbolUnavailable
	}
	var g uint64
	if err := check("cuMemGetAllocationGranularity", d.CuMemGetAllocationGranularity(&g, prop, option)); err != nil {
		return 0, err
	}
	return g, nil
}

// MemCreate creates a physical memory allocation of size bytes described by prop
// and returns its handle.
func MemCreate(d *cudasys.Driver, size uint64, prop *cudasys.CUmemAllocationProp) (cudasys.CUmemGenericAllocationHandle, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuMemCreate == nil {
		return 0, ErrSymbolUnavailable
	}
	var h cudasys.CUmemGenericAllocationHandle
	if err := check("cuMemCreate", d.CuMemCreate(&h, size, prop, 0)); err != nil {
		return 0, err
	}
	return h, nil
}

// MemAddressReserve reserves a virtual address range of size bytes with the
// given alignment and returns its base pointer.
func MemAddressReserve(d *cudasys.Driver, size, alignment uint64) (cudasys.CUdeviceptr, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuMemAddressReserve == nil {
		return 0, ErrSymbolUnavailable
	}
	var ptr cudasys.CUdeviceptr
	if err := check("cuMemAddressReserve", d.CuMemAddressReserve(&ptr, size, alignment, 0, 0)); err != nil {
		return 0, err
	}
	return ptr, nil
}

// MemMap maps the physical allocation handle into the reserved range at ptr.
func MemMap(d *cudasys.Driver, ptr cudasys.CUdeviceptr, size uint64, handle cudasys.CUmemGenericAllocationHandle) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemMap == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemMap", d.CuMemMap(ptr, size, 0, handle, 0))
}

// MemSetAccess applies the access description desc to the mapping at ptr.
func MemSetAccess(d *cudasys.Driver, ptr cudasys.CUdeviceptr, size uint64, desc *cudasys.CUmemAccessDesc) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemSetAccess == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemSetAccess", d.CuMemSetAccess(ptr, size, desc, 1))
}

// MemUnmap unmaps the range of size bytes at ptr.
func MemUnmap(d *cudasys.Driver, ptr cudasys.CUdeviceptr, size uint64) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemUnmap == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemUnmap", d.CuMemUnmap(ptr, size))
}

// MemAddressFree frees the reserved address range of size bytes at ptr.
func MemAddressFree(d *cudasys.Driver, ptr cudasys.CUdeviceptr, size uint64) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemAddressFree == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemAddressFree", d.CuMemAddressFree(ptr, size))
}

// MemRelease releases the physical allocation handle.
func MemRelease(d *cudasys.Driver, handle cudasys.CUmemGenericAllocationHandle) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemRelease == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemRelease", d.CuMemRelease(handle))
}
