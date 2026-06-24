package cudasys

// Memcpy2D mirrors the CUDA driver CUDA_MEMCPY2D structure passed to cuMemcpy2D.
// Field order and the padding after each memory-type field match the C layout
// on a 64-bit ABI; the size is guarded by a test.
type Memcpy2D struct {
	SrcXInBytes   uint64
	SrcY          uint64
	SrcMemoryType uint32
	_             uint32
	SrcHost       uintptr
	SrcDevice     CUdeviceptr
	SrcArray      uintptr
	SrcPitch      uint64
	DstXInBytes   uint64
	DstY          uint64
	DstMemoryType uint32
	_             uint32
	DstHost       uintptr
	DstDevice     CUdeviceptr
	DstArray      uintptr
	DstPitch      uint64
	WidthInBytes  uint64
	Height        uint64
}

// CUDA memory types for the Memcpy2D memory-type fields.
const (
	MemoryTypeHost    uint32 = 1
	MemoryTypeDevice  uint32 = 2
	MemoryTypeArray   uint32 = 3
	MemoryTypeUnified uint32 = 4
)
