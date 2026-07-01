package cudasys

import "unsafe"

// Memcpy3D mirrors the CUDA driver CUDA_MEMCPY3D structure passed to cuMemcpy3D.
// Field order, the padding after each memory-type field, and the two reserved
// pointers match the C layout on a 64-bit ABI; the size is guarded by a test.
// The host fields are unsafe.Pointer rather than uintptr so the garbage
// collector keeps the referenced host memory alive while a synchronous Memcpy3D
// runs. Memcpy3DAsync returns once the copy is enqueued, so its caller must keep
// the host memory alive until the stream completes. The reserved fields must be
// left zero.
type Memcpy3D struct {
	SrcXInBytes   uint64
	SrcY          uint64
	SrcZ          uint64
	SrcLOD        uint64
	SrcMemoryType uint32
	_             uint32
	SrcHost       unsafe.Pointer
	SrcDevice     CUdeviceptr
	SrcArray      uintptr
	_             uintptr
	SrcPitch      uint64
	SrcHeight     uint64
	DstXInBytes   uint64
	DstY          uint64
	DstZ          uint64
	DstLOD        uint64
	DstMemoryType uint32
	_             uint32
	DstHost       unsafe.Pointer
	DstDevice     CUdeviceptr
	DstArray      uintptr
	_             uintptr
	DstPitch      uint64
	DstHeight     uint64
	WidthInBytes  uint64
	Height        uint64
	Depth         uint64
}
