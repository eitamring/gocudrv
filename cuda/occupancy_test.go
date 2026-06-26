package cuda

import (
	"errors"
	"math"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func occupancyTestFunction(t *testing.T) (*Context, *Function) {
	t.Helper()
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)
	mod, err := ctx.LoadModule([]byte{'P', 'T', 'X', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	t.Cleanup(func() { _ = mod.Close() })
	fn, err := mod.Function("k")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	return ctx, fn
}

func TestMaxActiveBlocksPerSM(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	var gotBlock int32
	var gotDyn uint64
	ctx.driver.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(n *int32, _ cudasys.CUfunction, blockSize int32, dyn uint64) cudasys.CUresult {
		gotBlock, gotDyn = blockSize, dyn
		*n = 6
		return cudasys.CUDA_SUCCESS
	}
	n, err := fn.MaxActiveBlocksPerSM(256, 1024)
	if err != nil {
		t.Fatalf("MaxActiveBlocksPerSM: %v", err)
	}
	if n != 6 {
		t.Errorf("n = %d, want 6", n)
	}
	if gotBlock != 256 || gotDyn != 1024 {
		t.Errorf("driver got blockSize=%d dyn=%d, want 256, 1024", gotBlock, gotDyn)
	}
}

func TestMaxActiveBlocksPerSMRejects(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	ctx.driver.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(*int32, cudasys.CUfunction, int32, uint64) cudasys.CUresult {
		t.Error("driver must not be called on rejected input")
		return cudasys.CUDA_SUCCESS
	}
	cases := []struct {
		name string
		fn   func() error
		want error
	}{
		{"nil function", func() error { var f *Function; _, e := f.MaxActiveBlocksPerSM(256, 0); return e }, ErrNilFunction},
		{"zero block size", func() error { _, e := fn.MaxActiveBlocksPerSM(0, 0); return e }, ErrInvalidBlockSize},
		{"negative block size", func() error { _, e := fn.MaxActiveBlocksPerSM(-1, 0); return e }, ErrInvalidBlockSize},
		{"block size overflows int32", func() error { _, e := fn.MaxActiveBlocksPerSM(math.MaxInt32+1, 0); return e }, ErrInvalidBlockSize},
		{"negative dynamic shared", func() error { _, e := fn.MaxActiveBlocksPerSM(256, -1); return e }, ErrInvalidLength},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestMaxActiveBlocksPerSMPropagatesError(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	ctx.driver.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(*int32, cudasys.CUfunction, int32, uint64) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if _, err := fn.MaxActiveBlocksPerSM(256, 0); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}

func TestSuggestedBlockSize(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	var gotLimit int32
	ctx.driver.CuOccupancyMaxPotentialBlockSize = func(minGrid *int32, block *int32, _ cudasys.CUfunction, b2d uintptr, _ uint64, limit int32) cudasys.CUresult {
		if b2d != 0 {
			t.Errorf("block-size-to-shared-mem callback = %#x, want 0 (null)", b2d)
		}
		gotLimit = limit
		*minGrid, *block = 80, 256
		return cudasys.CUDA_SUCCESS
	}
	minGrid, block, err := fn.SuggestedBlockSize(0, 512)
	if err != nil {
		t.Fatalf("SuggestedBlockSize: %v", err)
	}
	if minGrid != 80 || block != 256 {
		t.Errorf("got minGrid=%d block=%d, want 80, 256", minGrid, block)
	}
	if gotLimit != 512 {
		t.Errorf("blockSizeLimit = %d, want 512", gotLimit)
	}
}

func TestSuggestedBlockSizeRejects(t *testing.T) {
	_, fn := occupancyTestFunction(t)
	if _, _, err := fn.SuggestedBlockSize(-1, 0); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("negative shared err = %v, want ErrInvalidLength", err)
	}
	if _, _, err := fn.SuggestedBlockSize(0, math.MaxInt32+1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("block-size-limit overflow err = %v, want ErrInvalidLength", err)
	}
	var nilFn *Function
	if _, _, err := nilFn.SuggestedBlockSize(0, 0); !errors.Is(err, ErrNilFunction) {
		t.Errorf("nil function err = %v, want ErrNilFunction", err)
	}
}

func TestSuggestedConfig1D(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	ctx.driver.CuOccupancyMaxPotentialBlockSize = func(minGrid *int32, block *int32, _ cudasys.CUfunction, _ uintptr, _ uint64, _ int32) cudasys.CUresult {
		*minGrid, *block = 40, 128
		return cudasys.CUDA_SUCCESS
	}
	cfg, err := fn.SuggestedConfig1D(1000, 256)
	if err != nil {
		t.Fatalf("SuggestedConfig1D: %v", err)
	}
	if cfg.BlockX != 128 {
		t.Errorf("BlockX = %d, want 128", cfg.BlockX)
	}
	if cfg.GridX != 8 { // ceil(1000/128) = 8
		t.Errorf("GridX = %d, want 8", cfg.GridX)
	}
	if cfg.SharedMemBytes != 256 {
		t.Errorf("SharedMemBytes = %d, want 256", cfg.SharedMemBytes)
	}
	if !cfg.valid() {
		t.Error("config not valid")
	}
}

func TestSuggestedConfig1DRejects(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	if _, err := fn.SuggestedConfig1D(0, 0); !errors.Is(err, ErrInvalidLaunchConfig) {
		t.Errorf("n=0 err = %v, want ErrInvalidLaunchConfig", err)
	}
	ctx.driver.CuOccupancyMaxPotentialBlockSize = func(*int32, *int32, cudasys.CUfunction, uintptr, uint64, int32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if _, err := fn.SuggestedConfig1D(1000, 0); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("driver error not propagated: %v", err)
	}
}

func TestSuggestedBlockSizePropagatesError(t *testing.T) {
	ctx, fn := occupancyTestFunction(t)
	ctx.driver.CuOccupancyMaxPotentialBlockSize = func(*int32, *int32, cudasys.CUfunction, uintptr, uint64, int32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if _, _, err := fn.SuggestedBlockSize(0, 0); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}

func TestOccupancyAfterModuleClose(t *testing.T) {
	var f moduleFake
	ctx := newModuleTestContext(t, &f, nil)
	mod, err := ctx.LoadModule([]byte{'P', 'T', 'X', 0})
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("k")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	ctx.driver.CuOccupancyMaxActiveBlocksPerMultiprocessor = func(*int32, cudasys.CUfunction, int32, uint64) cudasys.CUresult {
		t.Error("MaxActiveBlocksPerSM driver must not run after module close")
		return cudasys.CUDA_SUCCESS
	}
	ctx.driver.CuOccupancyMaxPotentialBlockSize = func(*int32, *int32, cudasys.CUfunction, uintptr, uint64, int32) cudasys.CUresult {
		t.Error("SuggestedBlockSize driver must not run after module close")
		return cudasys.CUDA_SUCCESS
	}
	if err := mod.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := fn.MaxActiveBlocksPerSM(256, 0); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("MaxActiveBlocksPerSM after close = %v, want ErrModuleClosed", err)
	}
	if _, _, err := fn.SuggestedBlockSize(0, 0); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("SuggestedBlockSize after close = %v, want ErrModuleClosed", err)
	}
}
