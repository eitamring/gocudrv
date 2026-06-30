package cudasys

import "unsafe"

// CUmemGenericAllocationHandle is an opaque handle to a physical VMM allocation
// returned by cuMemCreate.
type CUmemGenericAllocationHandle uint64

// CUmemLocation identifies a memory location (a device or the host). It mirrors
// the CUmemLocation C struct.
type CUmemLocation struct {
	Type int32 // CUmemLocationType
	Id   int32
}

// CUmemAllocationFlags mirrors the allocFlags member of CUmemAllocationProp.
type CUmemAllocationFlags struct {
	CompressionType      uint8
	GpuDirectRDMACapable uint8
	Usage                uint16
	Reserved             [4]uint8
}

// CUmemAllocationProp describes a physical allocation for cuMemCreate. Field
// order, the trailing pointer alignment, and the size match the C
// CUmemAllocationProp on a 64-bit ABI; the layout is guarded by a test.
type CUmemAllocationProp struct {
	Type                 int32 // CUmemAllocationType
	RequestedHandleTypes int32 // CUmemAllocationHandleType
	Location             CUmemLocation
	Win32HandleMetaData  unsafe.Pointer
	AllocFlags           CUmemAllocationFlags
}

// CUmemAccessDesc describes the access rights to grant a mapping for
// cuMemSetAccess. It mirrors the C CUmemAccessDesc.
type CUmemAccessDesc struct {
	Location CUmemLocation
	Flags    int32 // CUmemAccess_flags
}
