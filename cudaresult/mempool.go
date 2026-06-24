package cudaresult

import (
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// DeviceGetDefaultMemPool returns the device's default memory pool. The symbol
// is bound best-effort, so this returns ErrSymbolUnavailable on a driver
// without stream-ordered memory pools.
func DeviceGetDefaultMemPool(d *cudasys.Driver, dev cudasys.CUdevice) (cudasys.CUmemoryPool, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuDeviceGetDefaultMemPool == nil {
		return 0, ErrSymbolUnavailable
	}
	var pool cudasys.CUmemoryPool
	if err := check("cuDeviceGetDefaultMemPool", d.CuDeviceGetDefaultMemPool(&pool, dev)); err != nil {
		return 0, err
	}
	return pool, nil
}

// MemPoolGetAttributeU64 reads a uint64-valued pool attribute (the release
// threshold and the reserved/used memory counters).
func MemPoolGetAttributeU64(d *cudasys.Driver, pool cudasys.CUmemoryPool, attr int32) (uint64, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuMemPoolGetAttribute == nil {
		return 0, ErrSymbolUnavailable
	}
	var v uint64
	if err := check("cuMemPoolGetAttribute", d.CuMemPoolGetAttribute(pool, attr, unsafe.Pointer(&v))); err != nil {
		return 0, err
	}
	return v, nil
}

// MemPoolSetAttributeU64 writes a uint64-valued pool attribute.
func MemPoolSetAttributeU64(d *cudasys.Driver, pool cudasys.CUmemoryPool, attr int32, value uint64) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemPoolSetAttribute == nil {
		return ErrSymbolUnavailable
	}
	v := value
	return check("cuMemPoolSetAttribute", d.CuMemPoolSetAttribute(pool, attr, unsafe.Pointer(&v)))
}

// MemAllocFromPoolAsync allocates bytes from pool, ordered on stream, and
// returns the device pointer after the driver accepts the work.
func MemAllocFromPoolAsync(d *cudasys.Driver, bytes uint64, pool cudasys.CUmemoryPool, stream cudasys.CUstream) (cudasys.CUdeviceptr, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuMemAllocFromPoolAsync == nil {
		return 0, ErrSymbolUnavailable
	}
	var ptr cudasys.CUdeviceptr
	if err := check("cuMemAllocFromPoolAsync", d.CuMemAllocFromPoolAsync(&ptr, bytes, pool, stream)); err != nil {
		return 0, err
	}
	return ptr, nil
}
