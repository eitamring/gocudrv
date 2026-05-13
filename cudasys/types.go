package cudasys

// Driver API handle and result types. Layout matches the public CUDA header
// so values can be passed across the dynamic-call boundary without
// conversion.
type (
	CUresult    int32
	CUdevice    int32
	CUcontext   uintptr
	CUstream    uintptr
	CUmodule    uintptr
	CUfunction  uintptr
	CUdeviceptr uint64
	CUevent     uintptr
)
