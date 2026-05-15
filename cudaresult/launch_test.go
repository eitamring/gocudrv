package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestLaunchKernel(t *testing.T) {
	value := uint32(7)
	params := []unsafe.Pointer{unsafe.Pointer(&value)}

	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuLaunchKernel: func(
				fn cudasys.CUfunction,
				gridX, gridY, gridZ uint32,
				blockX, blockY, blockZ uint32,
				sharedMemBytes uint32,
				stream cudasys.CUstream,
				gotParams *unsafe.Pointer,
				extra *unsafe.Pointer,
			) cudasys.CUresult {
				if fn != 0xCAFE {
					t.Errorf("fn = %#x, want 0xCAFE", fn)
				}
				if gridX != 2 || gridY != 3 || gridZ != 4 {
					t.Errorf("grid = (%d,%d,%d), want (2,3,4)", gridX, gridY, gridZ)
				}
				if blockX != 5 || blockY != 6 || blockZ != 7 {
					t.Errorf("block = (%d,%d,%d), want (5,6,7)", blockX, blockY, blockZ)
				}
				if sharedMemBytes != 8 {
					t.Errorf("shared = %d, want 8", sharedMemBytes)
				}
				if stream != 9 {
					t.Errorf("stream = %#x, want 9", stream)
				}
				if extra != nil {
					t.Errorf("extra = %p, want nil", extra)
				}
				if gotParams == nil {
					t.Fatal("nil kernel params")
				}
				got := unsafe.Slice(gotParams, 1)
				if *(*uint32)(got[0]) != value {
					t.Errorf("arg0 = %d, want %d", *(*uint32)(got[0]), value)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"launch out of resources",
			&cudasys.Driver{CuLaunchKernel: func(
				cudasys.CUfunction,
				uint32, uint32, uint32,
				uint32, uint32, uint32,
				uint32,
				cudasys.CUstream,
				*unsafe.Pointer,
				*unsafe.Pointer,
			) cudasys.CUresult {
				return cudasys.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
			}},
			ErrLaunchOutOfResources,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := LaunchKernel(tc.driver, 0xCAFE, 2, 3, 4, 5, 6, 7, 8, 9, &params[0])
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
