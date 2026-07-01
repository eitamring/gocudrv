package cuda

import "context"

// LaunchCooperative launches f as a cooperative kernel on the context's legacy
// default stream. Its blocks can synchronize as a whole grid, which requires
// every block to be co-resident: size the grid with MaxCooperativeGridBlocks.
// Returns ErrSymbolUnavailable on a driver without cuLaunchCooperativeKernel;
// argument-lifetime and cancellation rules match Launch.
func (f *Function) LaunchCooperative(ctx context.Context, cfg LaunchConfig, args ...KernelArg) error {
	return f.launch(ctx, defaultStream, nil, cfg, true, args...)
}

// LaunchCooperativeOn is LaunchCooperative on a specific stream of the same
// Context.
func (f *Function) LaunchCooperativeOn(ctx context.Context, stream *Stream, cfg LaunchConfig, args ...KernelArg) error {
	if f == nil {
		return ErrNilFunction
	}
	if stream == nil {
		return ErrNilStream
	}
	stream.opMu.RLock()
	defer stream.opMu.RUnlock()
	if stream.closed {
		return ErrStreamClosed
	}
	return f.launch(ctx, stream.raw, stream.ctx, cfg, true, args...)
}

// MaxCooperativeGridBlocks returns the largest total block count a cooperative
// launch of f may use while keeping every block co-resident, which a grid-wide
// barrier requires: the multiprocessor count times MaxActiveBlocksPerSM.
func (f *Function) MaxCooperativeGridBlocks(blockSize, dynamicSharedMem int) (int, error) {
	perSM, err := f.MaxActiveBlocksPerSM(blockSize, dynamicSharedMem)
	if err != nil {
		return 0, err
	}
	sm, err := f.module.ctx.Device().Attribute(DeviceAttributeMultiprocessorCount)
	if err != nil {
		return 0, err
	}
	return perSM * sm, nil
}
