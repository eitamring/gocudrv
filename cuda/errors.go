package cuda

import "github.com/eitamring/gocudrv/cudaresult"

// Error is the typed error returned for non-success CUDA result codes.
// Compare with errors.Is against the sentinels below.
type Error = cudaresult.Error

// Sentinel errors covering the most common CUDA driver result codes.
var (
	ErrInvalidValue               = cudaresult.ErrInvalidValue
	ErrOutOfMemory                = cudaresult.ErrOutOfMemory
	ErrNotInitialized             = cudaresult.ErrNotInitialized
	ErrDeinitialized              = cudaresult.ErrDeinitialized
	ErrProfilerDisabled           = cudaresult.ErrProfilerDisabled
	ErrStubLibrary                = cudaresult.ErrStubLibrary
	ErrDeviceUnavailable          = cudaresult.ErrDeviceUnavailable
	ErrNoDevice                   = cudaresult.ErrNoDevice
	ErrInvalidDevice              = cudaresult.ErrInvalidDevice
	ErrDeviceNotLicensed          = cudaresult.ErrDeviceNotLicensed
	ErrInvalidImage               = cudaresult.ErrInvalidImage
	ErrInvalidContext             = cudaresult.ErrInvalidContext
	ErrNoBinaryForGPU             = cudaresult.ErrNoBinaryForGPU
	ErrInvalidPTX                 = cudaresult.ErrInvalidPTX
	ErrUnsupportedPTXVersion      = cudaresult.ErrUnsupportedPTXVersion
	ErrInvalidSource              = cudaresult.ErrInvalidSource
	ErrFileNotFound               = cudaresult.ErrFileNotFound
	ErrSharedObjectSymbolNotFound = cudaresult.ErrSharedObjectSymbolNotFound
	ErrSharedObjectInitFailed     = cudaresult.ErrSharedObjectInitFailed
	ErrOperatingSystem            = cudaresult.ErrOperatingSystem
	ErrInvalidHandle              = cudaresult.ErrInvalidHandle
	ErrIllegalState               = cudaresult.ErrIllegalState
	ErrNotFound                   = cudaresult.ErrNotFound
	ErrNotReady                   = cudaresult.ErrNotReady
	ErrIllegalAddress             = cudaresult.ErrIllegalAddress
	ErrLaunchOutOfResources       = cudaresult.ErrLaunchOutOfResources
	ErrLaunchTimeout              = cudaresult.ErrLaunchTimeout
	ErrLaunchFailed               = cudaresult.ErrLaunchFailed
	ErrNotPermitted               = cudaresult.ErrNotPermitted
	ErrNotSupported               = cudaresult.ErrNotSupported
	ErrSystemNotReady             = cudaresult.ErrSystemNotReady
	ErrSystemDriverMismatch       = cudaresult.ErrSystemDriverMismatch
	ErrUnknown                    = cudaresult.ErrUnknown
)
