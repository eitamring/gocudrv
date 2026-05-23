package cuda

import (
	"errors"

	"github.com/eitamring/gocudrv/cudaresult"
)

// Go-side sentinels that signal wrapper-level rejections separate from CUDA
// result codes. Use errors.Is to match.
var (
	ErrContextClosed         = errors.New("cuda: context is closed")
	ErrNilContext            = errors.New("cuda: nil context")
	ErrNilBuffer             = errors.New("cuda: nil buffer")
	ErrBufferClosed          = errors.New("cuda: buffer is closed")
	ErrLengthMismatch        = errors.New("cuda: length mismatch")
	ErrInvalidLength         = errors.New("cuda: invalid length")
	ErrNilModule             = errors.New("cuda: nil module")
	ErrModuleClosed          = errors.New("cuda: module is closed")
	ErrEmptyImage            = errors.New("cuda: empty module image")
	ErrEmptyFunctionName     = errors.New("cuda: empty function name")
	ErrInvalidFunctionName   = errors.New("cuda: function name contains null byte")
	ErrNilFunction           = errors.New("cuda: nil function")
	ErrNilStream             = errors.New("cuda: nil stream")
	ErrStreamClosed          = errors.New("cuda: stream is closed")
	ErrInvalidStreamPriority = errors.New("cuda: invalid stream priority")
	ErrNilEvent              = errors.New("cuda: nil event")
	ErrEventClosed           = errors.New("cuda: event is closed")
	ErrEventTimingDisabled   = errors.New("cuda: event timing is disabled")
	ErrInvalidLaunchConfig   = errors.New("cuda: invalid launch config")
	ErrNilKernelArg          = errors.New("cuda: nil kernel argument")
	ErrContextMismatch       = errors.New("cuda: resource belongs to a different context")
)

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
