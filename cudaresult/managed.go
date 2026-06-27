package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// MemAllocManaged allocates bytesize bytes of unified (managed) memory that is
// addressable from both host and device, returning the host-usable pointer.
// flags is a CU_MEM_ATTACH_* value.
func MemAllocManaged(d *cudasys.Driver, bytesize uint64, flags uint32) (*byte, error) {
	if d == nil {
		return nil, ErrNotInitialized
	}
	if d.CuMemAllocManaged == nil {
		return nil, ErrSymbolUnavailable
	}
	var p *byte
	if err := check("cuMemAllocManaged", d.CuMemAllocManaged(&p, bytesize, flags)); err != nil {
		return nil, err
	}
	return p, nil
}

// MemPrefetchAsync migrates count bytes at devPtr to dstDevice ahead of use,
// ordered on stream. dstDevice is a CUdevice, or CU_DEVICE_CPU for the host.
func MemPrefetchAsync(d *cudasys.Driver, devPtr cudasys.CUdeviceptr, count uint64, dstDevice cudasys.CUdevice, stream cudasys.CUstream) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemPrefetchAsync == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemPrefetchAsync", d.CuMemPrefetchAsync(devPtr, count, dstDevice, stream))
}

// MemAdvise applies a CU_MEM_ADVISE_* hint to count bytes at devPtr for device.
func MemAdvise(d *cudasys.Driver, devPtr cudasys.CUdeviceptr, count uint64, advice int32, device cudasys.CUdevice) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemAdvise == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemAdvise", d.CuMemAdvise(devPtr, count, advice, device))
}
