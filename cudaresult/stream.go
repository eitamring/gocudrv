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

// StreamWaitEvent makes stream wait until event has completed.
func StreamWaitEvent(d *cudasys.Driver, stream cudasys.CUstream, event cudasys.CUevent, flags uint32) error {
	if d == nil || d.CuStreamWaitEvent == nil {
		return ErrNotInitialized
	}
	return check("cuStreamWaitEvent", d.CuStreamWaitEvent(stream, event, flags))
}

// EventCreate creates a CUDA event with the supplied creation flags.
func EventCreate(d *cudasys.Driver, flags uint32) (cudasys.CUevent, error) {
	if d == nil || d.CuEventCreate == nil {
		return 0, ErrNotInitialized
	}
	var event cudasys.CUevent
	if err := check("cuEventCreate", d.CuEventCreate(&event, flags)); err != nil {
		return 0, err
	}
	return event, nil
}

// EventDestroy destroys an event previously returned by EventCreate.
func EventDestroy(d *cudasys.Driver, event cudasys.CUevent) error {
	if d == nil || d.CuEventDestroy == nil {
		return ErrNotInitialized
	}
	return check("cuEventDestroy_v2", d.CuEventDestroy(event))
}

// EventRecord records event into stream.
func EventRecord(d *cudasys.Driver, event cudasys.CUevent, stream cudasys.CUstream) error {
	if d == nil || d.CuEventRecord == nil {
		return ErrNotInitialized
	}
	return check("cuEventRecord", d.CuEventRecord(event, stream))
}

// EventQuery reports whether event has completed. It returns ErrNotReady if
// CUDA reports the event is still pending.
func EventQuery(d *cudasys.Driver, event cudasys.CUevent) error {
	if d == nil || d.CuEventQuery == nil {
		return ErrNotInitialized
	}
	return check("cuEventQuery", d.CuEventQuery(event))
}

// EventSynchronize blocks until event has completed.
func EventSynchronize(d *cudasys.Driver, event cudasys.CUevent) error {
	if d == nil || d.CuEventSynchronize == nil {
		return ErrNotInitialized
	}
	return check("cuEventSynchronize", d.CuEventSynchronize(event))
}

// EventElapsedTime returns milliseconds elapsed between two recorded events.
func EventElapsedTime(d *cudasys.Driver, start, end cudasys.CUevent) (float32, error) {
	if d == nil || d.CuEventElapsedTime == nil {
		return 0, ErrNotInitialized
	}
	var ms float32
	if err := check("cuEventElapsedTime", d.CuEventElapsedTime(&ms, start, end)); err != nil {
		return 0, err
	}
	return ms, nil
}
