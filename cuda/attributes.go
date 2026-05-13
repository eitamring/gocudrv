package cuda

// DeviceAttribute identifies a queryable integer property of a CUDA device.
// Numeric values match the CUDA driver header.
type DeviceAttribute int32

// Device attributes exposed in v0. The full enumeration in the CUDA header
// has many more entries; additions are mechanical when needed.
const (
	DeviceAttributeMaxThreadsPerBlock     DeviceAttribute = 1
	DeviceAttributeWarpSize               DeviceAttribute = 10
	DeviceAttributeClockRate              DeviceAttribute = 13
	DeviceAttributeMultiprocessorCount    DeviceAttribute = 16
	DeviceAttributeIntegrated             DeviceAttribute = 18
	DeviceAttributeConcurrentKernels      DeviceAttribute = 31
	DeviceAttributeMemoryClockRate        DeviceAttribute = 36
	DeviceAttributeGlobalMemoryBusWidth   DeviceAttribute = 37
	DeviceAttributeComputeCapabilityMajor DeviceAttribute = 75
	DeviceAttributeComputeCapabilityMinor DeviceAttribute = 76
)
