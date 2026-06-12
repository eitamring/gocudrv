package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestOccupancyMaxActiveBlocksPerMultiprocessor(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
		wantN   int
	}{
		{"nil driver", nil, ErrNotInitialized, 0},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized, 0},
		{
			"success",
			&cudasys.Driver{CuOccupancyMaxActiveBlocksPerMultiprocessor: func(n *int32, _ cudasys.CUfunction, blockSize int32, dyn uint64) cudasys.CUresult {
				if blockSize != 256 || dyn != 2048 {
					t.Errorf("got blockSize=%d dyn=%d", blockSize, dyn)
				}
				*n = 8
				return cudasys.CUDA_SUCCESS
			}},
			nil, 8,
		},
		{
			"invalid value",
			&cudasys.Driver{CuOccupancyMaxActiveBlocksPerMultiprocessor: func(*int32, cudasys.CUfunction, int32, uint64) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue, 0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, err := OccupancyMaxActiveBlocksPerMultiprocessor(tc.driver, 0xCAFE, 256, 2048)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if n != tc.wantN {
				t.Errorf("n = %d, want %d", n, tc.wantN)
			}
		})
	}
}

func TestOccupancyMaxPotentialBlockSize(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuOccupancyMaxPotentialBlockSize: func(minGrid *int32, block *int32, _ cudasys.CUfunction, b2d uintptr, dyn uint64, limit int32) cudasys.CUresult {
				if b2d != 0 {
					t.Errorf("callback = %#x, want 0 (null)", b2d)
				}
				if dyn != 1024 || limit != 512 {
					t.Errorf("got dyn=%d limit=%d", dyn, limit)
				}
				*minGrid, *block = 80, 256
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuOccupancyMaxPotentialBlockSize: func(*int32, *int32, cudasys.CUfunction, uintptr, uint64, int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			minGrid, block, err := OccupancyMaxPotentialBlockSize(tc.driver, 0xCAFE, 1024, 512)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if minGrid != 80 || block != 256 {
				t.Errorf("got minGrid=%d block=%d, want 80, 256", minGrid, block)
			}
		})
	}
}
