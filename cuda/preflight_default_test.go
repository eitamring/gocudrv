//go:build !cuda_integration

package cuda

// preflight is a no-op when the cuda_integration tag is not set. The
// integration-tagged build replaces this with a function that probes the
// real driver and skips the whole binary when CUDA is unusable.
func preflight() bool { return true }
