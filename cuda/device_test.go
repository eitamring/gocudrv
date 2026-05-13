package cuda

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func installDriver(t *testing.T, d *cudasys.Driver) {
	t.Helper()
	resetDriver()
	mu.Lock()
	driver = d
	mu.Unlock()
	t.Cleanup(resetDriver)
}

func TestDeviceCountBeforeInit(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)
	if _, err := DeviceCount(); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("err = %v, want ErrNotInitialized", err)
	}
}

func TestDeviceCount(t *testing.T) {
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(c *int32) cudasys.CUresult { *c = 2; return cudasys.CUDA_SUCCESS },
	})
	got, err := DeviceCount()
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != 2 {
		t.Errorf("count = %d, want 2", got)
	}
}

func TestGetDeviceBeforeInit(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)
	if _, err := GetDevice(0); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("err = %v, want ErrNotInitialized", err)
	}
}

func TestGetDeviceBounds(t *testing.T) {
	cases := []struct {
		name    string
		ordinal int
		count   int32
		wantErr error
	}{
		{"negative ordinal", -1, 2, ErrInvalidOrdinal},
		{"ordinal equals count", 2, 2, ErrInvalidOrdinal},
		{"ordinal above count", 5, 2, ErrInvalidOrdinal},
		{"zero count rejects zero ordinal", 0, 0, ErrInvalidOrdinal},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			installDriver(t, &cudasys.Driver{
				CuDeviceGetCount: func(c *int32) cudasys.CUresult { *c = tc.count; return cudasys.CUDA_SUCCESS },
				CuDeviceGet: func(*cudasys.CUdevice, int32) cudasys.CUresult {
					t.Error("cuDeviceGet must not be called when ordinal is rejected")
					return cudasys.CUDA_SUCCESS
				},
			})
			_, err := GetDevice(tc.ordinal)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestGetDeviceSuccess(t *testing.T) {
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(c *int32) cudasys.CUresult { *c = 4; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, ord int32) cudasys.CUresult {
			*dev = cudasys.CUdevice(ord * 10)
			return cudasys.CUDA_SUCCESS
		},
	})
	d, err := GetDevice(2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if d.Ordinal() != 2 {
		t.Errorf("ordinal = %d, want 2", d.Ordinal())
	}
	if d.handle != 20 {
		t.Errorf("handle = %v, want 20", d.handle)
	}
}

func TestDeviceMethods(t *testing.T) {
	major := int32(8)
	minor := int32(9)
	installDriver(t, &cudasys.Driver{
		CuDeviceGetCount: func(c *int32) cudasys.CUresult { *c = 1; return cudasys.CUDA_SUCCESS },
		CuDeviceGet: func(dev *cudasys.CUdevice, _ int32) cudasys.CUresult {
			*dev = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDeviceGetName: func(buf *byte, length int32, _ cudasys.CUdevice) cudasys.CUresult {
			name := "Test GPU"
			b := unsafeSlice(buf, int(length))
			copy(b, name)
			b[len(name)] = 0
			return cudasys.CUDA_SUCCESS
		},
		CuDeviceTotalMem: func(n *uint64, _ cudasys.CUdevice) cudasys.CUresult {
			*n = 24 * 1024 * 1024 * 1024
			return cudasys.CUDA_SUCCESS
		},
		CuDeviceGetAttribute: func(v *int32, attr int32, _ cudasys.CUdevice) cudasys.CUresult {
			switch DeviceAttribute(attr) {
			case DeviceAttributeComputeCapabilityMajor:
				*v = major
			case DeviceAttributeComputeCapabilityMinor:
				*v = minor
			case DeviceAttributeMultiprocessorCount:
				*v = 60
			default:
				*v = 0
			}
			return cudasys.CUDA_SUCCESS
		},
	})

	d, err := GetDevice(0)
	if err != nil {
		t.Fatalf("get: %v", err)
	}

	name, err := d.Name()
	if err != nil {
		t.Fatalf("name: %v", err)
	}
	if name != "Test GPU" {
		t.Errorf("name = %q, want %q", name, "Test GPU")
	}

	mem, err := d.TotalMemory()
	if err != nil {
		t.Fatalf("total memory: %v", err)
	}
	if mem != 24*1024*1024*1024 {
		t.Errorf("memory = %d, want 24 GiB", mem)
	}

	maj, min, err := d.ComputeCapability()
	if err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	if maj != int(major) || min != int(minor) {
		t.Errorf("cc = %d.%d, want %d.%d", maj, min, major, minor)
	}

	sm, err := d.Attribute(DeviceAttributeMultiprocessorCount)
	if err != nil {
		t.Fatalf("multiprocessor count: %v", err)
	}
	if sm != 60 {
		t.Errorf("multiprocessors = %d, want 60", sm)
	}
}

func TestDeviceMethodsBeforeInit(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)
	d := &Device{ordinal: 0, handle: 0}
	if _, err := d.Name(); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("Name err = %v, want ErrNotInitialized", err)
	}
	if _, err := d.TotalMemory(); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("TotalMemory err = %v, want ErrNotInitialized", err)
	}
	if _, _, err := d.ComputeCapability(); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("ComputeCapability err = %v, want ErrNotInitialized", err)
	}
	if _, err := d.Attribute(DeviceAttributeWarpSize); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("Attribute err = %v, want ErrNotInitialized", err)
	}
}

func TestNilDeviceMethodsAfterInit(t *testing.T) {
	installDriver(t, &cudasys.Driver{})
	var d *Device
	if got := d.Ordinal(); got != -1 {
		t.Errorf("Ordinal = %d, want -1", got)
	}
	if _, err := d.Name(); !errors.Is(err, ErrNilDevice) {
		t.Errorf("Name err = %v, want ErrNilDevice", err)
	}
	if _, err := d.TotalMemory(); !errors.Is(err, ErrNilDevice) {
		t.Errorf("TotalMemory err = %v, want ErrNilDevice", err)
	}
	if _, _, err := d.ComputeCapability(); !errors.Is(err, ErrNilDevice) {
		t.Errorf("ComputeCapability err = %v, want ErrNilDevice", err)
	}
	if _, err := d.Attribute(DeviceAttributeWarpSize); !errors.Is(err, ErrNilDevice) {
		t.Errorf("Attribute err = %v, want ErrNilDevice", err)
	}
}
