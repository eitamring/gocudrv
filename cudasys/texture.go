package cudasys

// CUarray is an opaque handle to a CUDA array (driver CUarray).
type CUarray uintptr

// CUtexObject is a texture object handle (driver CUtexObject, an unsigned long
// long passed to kernels as a cudaTextureObject_t).
type CUtexObject uint64

// CUDA_ARRAY_DESCRIPTOR mirrors the driver 2D array descriptor passed to
// cuArrayCreate. The size and offsets are guarded by a test.
type CUDA_ARRAY_DESCRIPTOR struct {
	Width       uint64
	Height      uint64
	Format      uint32
	NumChannels uint32
}

// CUDA_RESOURCE_DESC mirrors the driver CUDA_RESOURCE_DESC. Only the array case
// of the resource union is modeled: Handle is res.array.hArray, and the rest of
// the 128-byte union is reserved padding. Field offsets are guarded by a test.
type CUDA_RESOURCE_DESC struct {
	ResType uint32
	_       uint32
	Handle  CUarray
	_       [120]byte
	Flags   uint32
	_       uint32
}

// CUDA_TEXTURE_DESC mirrors the driver CUDA_TEXTURE_DESC passed to
// cuTexObjectCreate. The size and offsets are guarded by a test.
type CUDA_TEXTURE_DESC struct {
	AddressMode         [3]uint32
	FilterMode          uint32
	Flags               uint32
	MaxAnisotropy       uint32
	MipmapFilterMode    uint32
	MipmapLevelBias     float32
	MinMipmapLevelClamp float32
	MaxMipmapLevelClamp float32
	BorderColor         [4]float32
	_                   [12]int32
}

// CUarray_format values for the array-descriptor Format field.
const (
	AdFormatUnsignedInt8  uint32 = 0x01
	AdFormatUnsignedInt16 uint32 = 0x02
	AdFormatUnsignedInt32 uint32 = 0x03
	AdFormatSignedInt8    uint32 = 0x08
	AdFormatSignedInt16   uint32 = 0x09
	AdFormatSignedInt32   uint32 = 0x0a
	AdFormatHalf          uint32 = 0x10
	AdFormatFloat         uint32 = 0x20
)

// Texture addressing and filtering modes, texture flags, and the array resource
// type for the descriptors above.
const (
	AddressModeWrap   uint32 = 0
	AddressModeClamp  uint32 = 1
	AddressModeMirror uint32 = 2
	AddressModeBorder uint32 = 3

	FilterModePoint  uint32 = 0
	FilterModeLinear uint32 = 1

	TexReadAsInteger        uint32 = 1
	TexNormalizedCoordinate uint32 = 2

	ResourceTypeArray uint32 = 0
)
