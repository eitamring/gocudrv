package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestDeviceCount(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    int
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuDeviceGetCount: func(c *int32) cudasys.CUresult {
				*c = 4
				return cudasys.CUDA_SUCCESS
			}},
			4,
			nil,
		},
		{
			"no device",
			&cudasys.Driver{CuDeviceGetCount: func(*int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_NO_DEVICE
			}},
			0,
			ErrNoDevice,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, err := DeviceCount(tc.driver)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if n != tc.want {
				t.Errorf("count = %d, want %d", n, tc.want)
			}
		})
	}
}

func TestGetDevice(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		ordinal int
		want    cudasys.CUdevice
		wantErr error
	}{
		{"nil driver", nil, 0, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuDeviceGet: func(dev *cudasys.CUdevice, ord int32) cudasys.CUresult {
				if ord != 2 {
					t.Errorf("ordinal = %d, want 2", ord)
				}
				*dev = 42
				return cudasys.CUDA_SUCCESS
			}},
			2,
			42,
			nil,
		},
		{
			"invalid device",
			&cudasys.Driver{CuDeviceGet: func(*cudasys.CUdevice, int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_DEVICE
			}},
			99,
			0,
			ErrInvalidDevice,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, err := GetDevice(tc.driver, tc.ordinal)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if h != tc.want {
				t.Errorf("handle = %v, want %v", h, tc.want)
			}
		})
	}
}

func TestDeviceName(t *testing.T) {
	writeName := func(s string) func(buf *byte, length int32, dev cudasys.CUdevice) cudasys.CUresult {
		return func(buf *byte, length int32, _ cudasys.CUdevice) cudasys.CUresult {
			b := unsafeSliceFromPtr(buf, int(length))
			copy(b, s)
			if len(s) < len(b) {
				b[len(s)] = 0
			}
			return cudasys.CUDA_SUCCESS
		}
	}

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    string
		wantErr error
	}{
		{"nil driver", nil, "", ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, "", ErrNotInitialized},
		{
			"short name",
			&cudasys.Driver{CuDeviceGetName: writeName("RTX 4070 Ti")},
			"RTX 4070 Ti",
			nil,
		},
		{
			"empty name",
			&cudasys.Driver{CuDeviceGetName: writeName("")},
			"",
			nil,
		},
		{
			"error",
			&cudasys.Driver{CuDeviceGetName: func(*byte, int32, cudasys.CUdevice) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_DEVICE
			}},
			"",
			ErrInvalidDevice,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DeviceName(tc.driver, 0)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("name = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestDeviceTotalMem(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    uint64
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuDeviceTotalMem: func(n *uint64, _ cudasys.CUdevice) cudasys.CUresult {
				*n = 12 * 1024 * 1024 * 1024
				return cudasys.CUDA_SUCCESS
			}},
			12 * 1024 * 1024 * 1024,
			nil,
		},
		{
			"error",
			&cudasys.Driver{CuDeviceTotalMem: func(*uint64, cudasys.CUdevice) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_DEVICE
			}},
			0,
			ErrInvalidDevice,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DeviceTotalMem(tc.driver, 0)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("bytes = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestDeviceAttribute(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		attr    int32
		want    int
		wantErr error
	}{
		{"nil driver", nil, 75, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 75, 0, ErrNotInitialized},
		{
			"compute capability major",
			&cudasys.Driver{CuDeviceGetAttribute: func(v *int32, attr int32, _ cudasys.CUdevice) cudasys.CUresult {
				if attr != 75 {
					t.Errorf("attr = %d, want 75", attr)
				}
				*v = 8
				return cudasys.CUDA_SUCCESS
			}},
			75,
			8,
			nil,
		},
		{
			"error",
			&cudasys.Driver{CuDeviceGetAttribute: func(*int32, int32, cudasys.CUdevice) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			999,
			0,
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DeviceAttribute(tc.driver, tc.attr, 0)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("value = %d, want %d", got, tc.want)
			}
		})
	}
}
