package cuda

import (
	"errors"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// ErrInvalidOrdinal is returned when GetDevice rejects an ordinal before
// calling into the CUDA driver. CUDA's own invalid-device error covers cases
// the driver rejects after the call.
var (
	ErrInvalidOrdinal = errors.New("cuda: invalid device ordinal")
	ErrNilDevice      = errors.New("cuda: nil device")
)

// Device is an opaque handle to a CUDA device returned by GetDevice. Methods
// require Init to have succeeded.
type Device struct {
	ordinal int
	handle  cudasys.CUdevice
}

// DeviceCount returns the number of CUDA-capable devices visible to the
// driver. Returns ErrNotInitialized if Init has not been called.
func DeviceCount() (int, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return 0, ErrNotInitialized
	}
	return cudaresult.DeviceCount(driver)
}

// GetDevice returns the device at the given ordinal. The ordinal is checked
// against DeviceCount; out-of-range values return ErrInvalidOrdinal.
func GetDevice(ordinal int) (*Device, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return nil, ErrNotInitialized
	}
	n, err := cudaresult.DeviceCount(driver)
	if err != nil {
		return nil, err
	}
	if ordinal < 0 || ordinal >= n {
		return nil, ErrInvalidOrdinal
	}
	h, err := cudaresult.GetDevice(driver, ordinal)
	if err != nil {
		return nil, err
	}
	return &Device{ordinal: ordinal, handle: h}, nil
}

// Ordinal returns the index this Device was acquired with.
func (d *Device) Ordinal() int {
	if d == nil {
		return -1
	}
	return d.ordinal
}

// Name returns the device name as reported by cuDeviceGetName.
func (d *Device) Name() (string, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return "", ErrNotInitialized
	}
	if d == nil {
		return "", ErrNilDevice
	}
	return cudaresult.DeviceName(driver, d.handle)
}

// TotalMemory returns the total device memory in bytes.
func (d *Device) TotalMemory() (uint64, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return 0, ErrNotInitialized
	}
	if d == nil {
		return 0, ErrNilDevice
	}
	return cudaresult.DeviceTotalMem(driver, d.handle)
}

// ComputeCapability returns the device's compute capability major and minor
// numbers from two cuDeviceGetAttribute calls.
func (d *Device) ComputeCapability() (major, minor int, err error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return 0, 0, ErrNotInitialized
	}
	if d == nil {
		return 0, 0, ErrNilDevice
	}
	major, err = cudaresult.DeviceAttribute(driver, int32(DeviceAttributeComputeCapabilityMajor), d.handle)
	if err != nil {
		return 0, 0, err
	}
	minor, err = cudaresult.DeviceAttribute(driver, int32(DeviceAttributeComputeCapabilityMinor), d.handle)
	if err != nil {
		return 0, 0, err
	}
	return major, minor, nil
}

// Attribute returns a single integer device attribute. Use the
// DeviceAttribute constants or pass a raw value for attributes not yet
// exposed by name.
func (d *Device) Attribute(attr DeviceAttribute) (int, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return 0, ErrNotInitialized
	}
	if d == nil {
		return 0, ErrNilDevice
	}
	return cudaresult.DeviceAttribute(driver, int32(attr), d.handle)
}
