package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestInit(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		flags   uint32
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuInit: func(flags uint32) cudasys.CUresult {
				if flags != 5 {
					t.Errorf("flags = %d, want 5", flags)
				}
				return cudasys.CUDA_SUCCESS
			}},
			5,
			nil,
		},
		{
			"out of memory",
			&cudasys.Driver{CuInit: func(uint32) cudasys.CUresult { return cudasys.CUDA_ERROR_OUT_OF_MEMORY }},
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := Init(tc.driver, tc.flags)
			if tc.wantErr == nil {
				if err != nil {
					t.Fatalf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestDriverVersion(t *testing.T) {
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
			&cudasys.Driver{CuDriverGetVersion: func(v *int32) cudasys.CUresult {
				*v = 12030
				return cudasys.CUDA_SUCCESS
			}},
			12030,
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuDriverGetVersion: func(*int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			0,
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := DriverVersion(tc.driver)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if v != tc.want {
				t.Errorf("version = %d, want %d", v, tc.want)
			}
		})
	}
}
