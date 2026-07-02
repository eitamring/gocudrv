package cuda

// DeviceAttribute identifies a queryable integer property of a CUDA device.
// Numeric values match the CUDA driver header.
type DeviceAttribute int32

// Device attributes exposed in v0. The full enumeration in the CUDA header
// has many more entries; additions are mechanical when needed.
const (
	DeviceAttributeMaxThreadsPerBlock               DeviceAttribute = 1
	DeviceAttributeMaxBlockDimX                     DeviceAttribute = 2
	DeviceAttributeMaxBlockDimY                     DeviceAttribute = 3
	DeviceAttributeMaxBlockDimZ                     DeviceAttribute = 4
	DeviceAttributeMaxGridDimX                      DeviceAttribute = 5
	DeviceAttributeMaxGridDimY                      DeviceAttribute = 6
	DeviceAttributeMaxGridDimZ                      DeviceAttribute = 7
	DeviceAttributeMaxSharedMemoryPerBlock          DeviceAttribute = 8
	DeviceAttributeTotalConstantMemory              DeviceAttribute = 9
	DeviceAttributeWarpSize                         DeviceAttribute = 10
	DeviceAttributeMaxRegistersPerBlock             DeviceAttribute = 12
	DeviceAttributeClockRate                        DeviceAttribute = 13
	DeviceAttributeMultiprocessorCount              DeviceAttribute = 16
	DeviceAttributeIntegrated                       DeviceAttribute = 18
	DeviceAttributeCanMapHostMemory                 DeviceAttribute = 19
	DeviceAttributeComputeMode                      DeviceAttribute = 20
	DeviceAttributeConcurrentKernels                DeviceAttribute = 31
	DeviceAttributePCIBusID                         DeviceAttribute = 33
	DeviceAttributePCIDeviceID                      DeviceAttribute = 34
	DeviceAttributeTCCDriver                        DeviceAttribute = 35
	DeviceAttributeMemoryClockRate                  DeviceAttribute = 36
	DeviceAttributeGlobalMemoryBusWidth             DeviceAttribute = 37
	DeviceAttributeL2CacheSize                      DeviceAttribute = 38
	DeviceAttributeMaxThreadsPerMultiprocessor      DeviceAttribute = 39
	DeviceAttributeAsyncEngineCount                 DeviceAttribute = 40
	DeviceAttributeUnifiedAddressing                DeviceAttribute = 41
	DeviceAttributePCIDomainID                      DeviceAttribute = 50
	DeviceAttributeComputeCapabilityMajor           DeviceAttribute = 75
	DeviceAttributeComputeCapabilityMinor           DeviceAttribute = 76
	DeviceAttributeMaxSharedMemoryPerMultiprocessor DeviceAttribute = 81
	DeviceAttributeMaxRegistersPerMultiprocessor    DeviceAttribute = 82
	DeviceAttributeManagedMemory                    DeviceAttribute = 83
	DeviceAttributeConcurrentManagedAccess          DeviceAttribute = 89
	DeviceAttributeCooperativeLaunch                DeviceAttribute = 95
)
