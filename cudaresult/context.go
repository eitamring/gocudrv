package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// PrimaryCtxRetain retains the primary context on the given device and
// returns the opaque handle. Each Retain must be balanced with a Release.
func PrimaryCtxRetain(d *cudasys.Driver, dev cudasys.CUdevice) (cudasys.CUcontext, error) {
	if d == nil || d.CuDevicePrimaryCtxRetain == nil {
		return 0, ErrNotInitialized
	}
	var ctx cudasys.CUcontext
	if err := check("cuDevicePrimaryCtxRetain", d.CuDevicePrimaryCtxRetain(&ctx, dev)); err != nil {
		return 0, err
	}
	return ctx, nil
}

// PrimaryCtxRelease releases one retain count of the primary context for
// the given device.
func PrimaryCtxRelease(d *cudasys.Driver, dev cudasys.CUdevice) error {
	if d == nil || d.CuDevicePrimaryCtxRelease == nil {
		return ErrNotInitialized
	}
	return check("cuDevicePrimaryCtxRelease_v2", d.CuDevicePrimaryCtxRelease(dev))
}

// CtxSetCurrent makes ctx the current context for the calling OS thread.
// Pass the zero value to clear the current context.
func CtxSetCurrent(d *cudasys.Driver, ctx cudasys.CUcontext) error {
	if d == nil || d.CuCtxSetCurrent == nil {
		return ErrNotInitialized
	}
	return check("cuCtxSetCurrent", d.CuCtxSetCurrent(ctx))
}

// CtxGetCurrent returns the current context bound to the calling OS
// thread, or the zero value if no context is current.
func CtxGetCurrent(d *cudasys.Driver) (cudasys.CUcontext, error) {
	if d == nil || d.CuCtxGetCurrent == nil {
		return 0, ErrNotInitialized
	}
	var ctx cudasys.CUcontext
	if err := check("cuCtxGetCurrent", d.CuCtxGetCurrent(&ctx)); err != nil {
		return 0, err
	}
	return ctx, nil
}

// CtxSynchronize blocks until all preceding work in the current context's
// streams has completed.
func CtxSynchronize(d *cudasys.Driver) error {
	if d == nil || d.CuCtxSynchronize == nil {
		return ErrNotInitialized
	}
	return check("cuCtxSynchronize", d.CuCtxSynchronize())
}

// CtxGetStreamPriorityRange returns the least and greatest meaningful stream
// priorities for the current context. Lower numeric values represent greater
// priority.
func CtxGetStreamPriorityRange(d *cudasys.Driver) (least, greatest int32, err error) {
	if d == nil || d.CuCtxGetStreamPriorityRange == nil {
		return 0, 0, ErrNotInitialized
	}
	if err := check("cuCtxGetStreamPriorityRange", d.CuCtxGetStreamPriorityRange(&least, &greatest)); err != nil {
		return 0, 0, err
	}
	return least, greatest, nil
}
