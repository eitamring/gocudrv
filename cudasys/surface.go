package cudasys

// CUsurfObject is a surface object handle (driver CUsurfObject, an unsigned
// long long passed to kernels as a cudaSurfaceObject_t).
type CUsurfObject uint64

// CUDA_ARRAY3D_DESCRIPTOR mirrors the driver 3D array descriptor passed to
// cuArray3DCreate. Depth 0 creates a 2D array. The size and offsets are guarded
// by a test.
type CUDA_ARRAY3D_DESCRIPTOR struct {
	Width       uint64
	Height      uint64
	Depth       uint64
	Format      uint32
	NumChannels uint32
	Flags       uint32
	_           uint32
}

// ArraySurfaceLoadStore requests an array that surfaces can read and write
// (driver CUDA_ARRAY3D_SURFACE_LDST).
const ArraySurfaceLoadStore uint32 = 0x02
