package cuda

import (
	"context"
	"math"

	"github.com/eitamring/gocudrv/cudaresult"
)

// MaxActiveBlocksPerSM returns the maximum number of blocks that can be
// resident at once on a single streaming multiprocessor when this function
// runs with blockSize threads per block and dynamicSharedMem bytes of dynamic
// shared memory. It is the core occupancy metric: a kernel that fits more
// blocks per SM has more warps available to hide memory latency. blockSize
// must be positive and dynamicSharedMem must not be negative.
func (f *Function) MaxActiveBlocksPerSM(blockSize, dynamicSharedMem int) (int, error) {
	if f == nil {
		return 0, ErrNilFunction
	}
	if blockSize <= 0 || blockSize > math.MaxInt32 {
		return 0, ErrInvalidBlockSize
	}
	if dynamicSharedMem < 0 {
		return 0, ErrInvalidLength
	}
	if f.module == nil {
		return 0, ErrNilModule
	}
	f.module.opMu.RLock()
	defer f.module.opMu.RUnlock()
	if f.module.closed {
		return 0, ErrModuleClosed
	}
	var n int
	err := f.module.ctx.do(context.Background(), func() error {
		v, e := cudaresult.OccupancyMaxActiveBlocksPerMultiprocessor(f.module.ctx.driver, f.raw, int32(blockSize), uint64(dynamicSharedMem))
		if e != nil {
			return e
		}
		n = v
		return nil
	})
	if err != nil {
		return 0, err
	}
	return n, nil
}

// SuggestedBlockSize asks the driver for a block size that maximizes occupancy
// for this function, and the minimum grid size needed to fully occupy the
// device at that block size. dynamicSharedMem is the fixed dynamic shared
// memory per block. blockSizeLimit caps the suggested block size; pass 0 for no
// limit. dynamicSharedMem must not be negative and blockSizeLimit must not be
// negative.
func (f *Function) SuggestedBlockSize(dynamicSharedMem, blockSizeLimit int) (minGridSize, blockSize int, err error) {
	if f == nil {
		return 0, 0, ErrNilFunction
	}
	if dynamicSharedMem < 0 || blockSizeLimit < 0 || blockSizeLimit > math.MaxInt32 {
		return 0, 0, ErrInvalidLength
	}
	if f.module == nil {
		return 0, 0, ErrNilModule
	}
	f.module.opMu.RLock()
	defer f.module.opMu.RUnlock()
	if f.module.closed {
		return 0, 0, ErrModuleClosed
	}
	err = f.module.ctx.do(context.Background(), func() error {
		mgs, bs, e := cudaresult.OccupancyMaxPotentialBlockSize(f.module.ctx.driver, f.raw, uint64(dynamicSharedMem), int32(blockSizeLimit))
		if e != nil {
			return e
		}
		minGridSize, blockSize = mgs, bs
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	return minGridSize, blockSize, nil
}

// SuggestedConfig1D builds a one-dimensional launch config for n elements using
// the occupancy-maximizing block size from SuggestedBlockSize. It folds the two
// steps every caller would otherwise write by hand: ask the driver for the best
// block size, then size the grid to cover n. dynamicSharedMem is the fixed
// dynamic shared memory per block and is copied into the returned config.
func (f *Function) SuggestedConfig1D(n, dynamicSharedMem int) (LaunchConfig, error) {
	if n <= 0 {
		return LaunchConfig{}, ErrInvalidLaunchConfig
	}
	_, blockSize, err := f.SuggestedBlockSize(dynamicSharedMem, 0)
	if err != nil {
		return LaunchConfig{}, err
	}
	if blockSize <= 0 {
		return LaunchConfig{}, ErrInvalidLaunchConfig
	}
	cfg := LaunchConfig1D(n, blockSize)
	if !cfg.valid() {
		return LaunchConfig{}, ErrInvalidLaunchConfig
	}
	cfg.SharedMemBytes = uint32(dynamicSharedMem)
	return cfg, nil
}
