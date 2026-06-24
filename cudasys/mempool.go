package cudasys

// CUmemoryPool is an opaque CUDA memory pool handle.
type CUmemoryPool uintptr

// CUmemPool_attribute values for cuMemPoolGetAttribute and
// cuMemPoolSetAttribute. The release-threshold and the reserved/used counters
// carry a uint64; the reuse-policy attributes carry an int.
const (
	MemPoolAttrReuseFollowEventDependencies   int32 = 1
	MemPoolAttrReuseAllowOpportunistic        int32 = 2
	MemPoolAttrReuseAllowInternalDependencies int32 = 3
	MemPoolAttrReleaseThreshold               int32 = 4
	MemPoolAttrReservedMemCurrent             int32 = 5
	MemPoolAttrReservedMemHigh                int32 = 6
	MemPoolAttrUsedMemCurrent                 int32 = 7
	MemPoolAttrUsedMemHigh                    int32 = 8
)
