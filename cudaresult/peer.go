package cudaresult

import (
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

// DeviceCanAccessPeer reports whether dev can directly access memory allocated
// on peer.
func DeviceCanAccessPeer(d *cudasys.Driver, dev, peer cudasys.CUdevice) (bool, error) {
	if d == nil {
		return false, ErrNotInitialized
	}
	if d.CuDeviceCanAccessPeer == nil {
		return false, ErrSymbolUnavailable
	}
	var can int32
	if err := check("cuDeviceCanAccessPeer", d.CuDeviceCanAccessPeer(&can, dev, peer)); err != nil {
		return false, err
	}
	return can != 0, nil
}

// CtxEnablePeerAccess lets the calling context access memory in peerContext.
func CtxEnablePeerAccess(d *cudasys.Driver, peerContext cudasys.CUcontext) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuCtxEnablePeerAccess == nil {
		return ErrSymbolUnavailable
	}
	return check("cuCtxEnablePeerAccess", d.CuCtxEnablePeerAccess(peerContext, 0))
}

// CtxDisablePeerAccess undoes a prior CtxEnablePeerAccess for peerContext.
func CtxDisablePeerAccess(d *cudasys.Driver, peerContext cudasys.CUcontext) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuCtxDisablePeerAccess == nil {
		return ErrSymbolUnavailable
	}
	return check("cuCtxDisablePeerAccess", d.CuCtxDisablePeerAccess(peerContext))
}

// MemcpyPeer copies byteCount bytes from src in srcContext to dst in dstContext.
func MemcpyPeer(d *cudasys.Driver, dst cudasys.CUdeviceptr, dstContext cudasys.CUcontext, src cudasys.CUdeviceptr, srcContext cudasys.CUcontext, byteCount uint64) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuMemcpyPeer == nil {
		return ErrSymbolUnavailable
	}
	return check("cuMemcpyPeer", d.CuMemcpyPeer(dst, dstContext, src, srcContext, byteCount))
}

// PointerGetAttribute reads one attribute of a device pointer into the storage
// at data. attribute is a CUpointer_attribute value; data must point at storage
// of the type that attribute returns.
func PointerGetAttribute(d *cudasys.Driver, data unsafe.Pointer, attribute int32, ptr cudasys.CUdeviceptr) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuPointerGetAttribute == nil {
		return ErrSymbolUnavailable
	}
	return check("cuPointerGetAttribute", d.CuPointerGetAttribute(data, attribute, ptr))
}
