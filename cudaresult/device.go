package cudaresult

import (
	"bytes"

	"github.com/eitamring/gocudrv/cudasys"
)

const deviceNameBufferLen = 256

// DeviceCount calls cuDeviceGetCount through d.
func DeviceCount(d *cudasys.Driver) (int, error) {
	if d == nil || d.CuDeviceGetCount == nil {
		return 0, ErrNotInitialized
	}
	var n int32
	if err := check("cuDeviceGetCount", d.CuDeviceGetCount(&n)); err != nil {
		return 0, err
	}
	return int(n), nil
}

// GetDevice calls cuDeviceGet through d and returns the opaque handle.
// Callers are expected to have validated the ordinal range.
func GetDevice(d *cudasys.Driver, ordinal int) (cudasys.CUdevice, error) {
	if d == nil || d.CuDeviceGet == nil {
		return 0, ErrNotInitialized
	}
	var dev cudasys.CUdevice
	if err := check("cuDeviceGet", d.CuDeviceGet(&dev, int32(ordinal))); err != nil {
		return 0, err
	}
	return dev, nil
}

// DeviceName fetches the device name with a fixed-size buffer and trims at
// the first null byte. The CUDA driver guarantees the name is shorter than
// the buffer for current hardware.
func DeviceName(d *cudasys.Driver, dev cudasys.CUdevice) (string, error) {
	if d == nil || d.CuDeviceGetName == nil {
		return "", ErrNotInitialized
	}
	buf := make([]byte, deviceNameBufferLen)
	if err := check("cuDeviceGetName", d.CuDeviceGetName(&buf[0], int32(len(buf)), dev)); err != nil {
		return "", err
	}
	if i := bytes.IndexByte(buf, 0); i >= 0 {
		return string(buf[:i]), nil
	}
	return string(buf), nil
}

// DeviceTotalMem returns the total amount of memory on the device in bytes.
func DeviceTotalMem(d *cudasys.Driver, dev cudasys.CUdevice) (uint64, error) {
	if d == nil || d.CuDeviceTotalMem == nil {
		return 0, ErrNotInitialized
	}
	var n uint64
	if err := check("cuDeviceTotalMem_v2", d.CuDeviceTotalMem(&n, dev)); err != nil {
		return 0, err
	}
	return n, nil
}

// DeviceAttribute returns a single integer attribute for the device.
func DeviceAttribute(d *cudasys.Driver, attr int32, dev cudasys.CUdevice) (int, error) {
	if d == nil || d.CuDeviceGetAttribute == nil {
		return 0, ErrNotInitialized
	}
	var v int32
	if err := check("cuDeviceGetAttribute", d.CuDeviceGetAttribute(&v, attr, dev)); err != nil {
		return 0, err
	}
	return int(v), nil
}
