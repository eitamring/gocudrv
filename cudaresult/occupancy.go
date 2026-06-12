package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// OccupancyMaxActiveBlocksPerMultiprocessor returns the maximum number of
// active blocks per multiprocessor for fn at blockSize threads and
// dynamicSMemSize bytes of dynamic shared memory.
func OccupancyMaxActiveBlocksPerMultiprocessor(d *cudasys.Driver, fn cudasys.CUfunction, blockSize int32, dynamicSMemSize uint64) (int, error) {
	if d == nil || d.CuOccupancyMaxActiveBlocksPerMultiprocessor == nil {
		return 0, ErrNotInitialized
	}
	var numBlocks int32
	if err := check("cuOccupancyMaxActiveBlocksPerMultiprocessor", d.CuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, fn, blockSize, dynamicSMemSize)); err != nil {
		return 0, err
	}
	return int(numBlocks), nil
}

// OccupancyMaxPotentialBlockSize returns a block size that maximizes occupancy
// for fn and the minimum grid size needed to reach it. blockSizeLimit caps the
// suggested block size; pass 0 for no limit. The dynamic-shared-memory size
// callback is always null, so dynamicSMemSize is treated as a fixed amount.
func OccupancyMaxPotentialBlockSize(d *cudasys.Driver, fn cudasys.CUfunction, dynamicSMemSize uint64, blockSizeLimit int32) (minGridSize, blockSize int, err error) {
	if d == nil || d.CuOccupancyMaxPotentialBlockSize == nil {
		return 0, 0, ErrNotInitialized
	}
	var mgs, bs int32
	if e := check("cuOccupancyMaxPotentialBlockSize", d.CuOccupancyMaxPotentialBlockSize(&mgs, &bs, fn, 0, dynamicSMemSize, blockSizeLimit)); e != nil {
		return 0, 0, e
	}
	return int(mgs), int(bs), nil
}
