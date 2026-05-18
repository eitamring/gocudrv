package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// StreamCreate creates a CUDA stream with the supplied creation flags.
func StreamCreate(d *cudasys.Driver, flags uint32) (cudasys.CUstream, error) {
	if d == nil || d.CuStreamCreate == nil {
		return 0, ErrNotInitialized
	}
	var stream cudasys.CUstream
	if err := check("cuStreamCreate", d.CuStreamCreate(&stream, flags)); err != nil {
		return 0, err
	}
	return stream, nil
}

// StreamCreateWithPriority creates a CUDA stream with the supplied creation
// flags and scheduling priority.
func StreamCreateWithPriority(d *cudasys.Driver, flags uint32, priority int32) (cudasys.CUstream, error) {
	if d == nil || d.CuStreamCreateWithPriority == nil {
		return 0, ErrNotInitialized
	}
	var stream cudasys.CUstream
	if err := check("cuStreamCreateWithPriority", d.CuStreamCreateWithPriority(&stream, flags, priority)); err != nil {
		return 0, err
	}
	return stream, nil
}

// StreamDestroy destroys a stream previously returned by StreamCreate.
func StreamDestroy(d *cudasys.Driver, stream cudasys.CUstream) error {
	if d == nil || d.CuStreamDestroy == nil {
		return ErrNotInitialized
	}
	return check("cuStreamDestroy_v2", d.CuStreamDestroy(stream))
}

// StreamSynchronize blocks until all preceding work in stream has completed.
func StreamSynchronize(d *cudasys.Driver, stream cudasys.CUstream) error {
	if d == nil || d.CuStreamSynchronize == nil {
		return ErrNotInitialized
	}
	return check("cuStreamSynchronize", d.CuStreamSynchronize(stream))
}
