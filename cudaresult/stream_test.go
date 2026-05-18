package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestStreamCreate(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUstream
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamCreate: func(stream *cudasys.CUstream, flags uint32) cudasys.CUresult {
				if flags != 1 {
					t.Errorf("flags = %d, want 1", flags)
				}
				*stream = 0x5151
				return cudasys.CUDA_SUCCESS
			}},
			0x5151,
			nil,
		},
		{
			"out of memory",
			&cudasys.Driver{CuStreamCreate: func(*cudasys.CUstream, uint32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StreamCreate(tc.driver, 1)
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
				t.Errorf("stream = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestStreamCreateWithPriority(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUstream
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamCreateWithPriority: func(stream *cudasys.CUstream, flags uint32, priority int32) cudasys.CUresult {
				if flags != 1 {
					t.Errorf("flags = %d, want 1", flags)
				}
				if priority != -1 {
					t.Errorf("priority = %d, want -1", priority)
				}
				*stream = 0x6161
				return cudasys.CUDA_SUCCESS
			}},
			0x6161,
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuStreamCreateWithPriority: func(*cudasys.CUstream, uint32, int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			0,
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StreamCreateWithPriority(tc.driver, 1, -1)
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
				t.Errorf("stream = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestStreamDestroy(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamDestroy: func(stream cudasys.CUstream) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuStreamDestroy: func(cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := StreamDestroy(tc.driver, 0x5151)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestStreamSynchronize(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamSynchronize: func(stream cudasys.CUstream) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuStreamSynchronize: func(cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := StreamSynchronize(tc.driver, 0x5151)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}
