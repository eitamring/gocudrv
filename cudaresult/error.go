package cudaresult

import (
	"fmt"

	"github.com/eitamring/gocudrv/cudasys"
)

// Error wraps a CUDA driver result code into a Go error. Op records the name
// of the driver call that produced the code.
type Error struct {
	Code cudasys.CUresult
	Op   string
}

func (e *Error) Error() string {
	name := e.Code.Name()
	if name == "" {
		name = fmt.Sprintf("CUDA_ERROR_%d", int32(e.Code))
	}
	if e.Op == "" {
		return name
	}
	return e.Op + ": " + name
}

// Is matches on result code, ignoring Op, so a wrapped Error with extra
// context still satisfies errors.Is against the bare sentinel.
func (e *Error) Is(target error) bool {
	other, ok := target.(*Error)
	if !ok {
		return false
	}
	return e.Code == other.Code
}

// Sentinel errors for the most common CUDA driver result codes. Use with
// errors.Is to match returned errors regardless of the Op field.
var (
	ErrInvalidValue               = &Error{Code: cudasys.CUDA_ERROR_INVALID_VALUE}
	ErrOutOfMemory                = &Error{Code: cudasys.CUDA_ERROR_OUT_OF_MEMORY}
	ErrNotInitialized             = &Error{Code: cudasys.CUDA_ERROR_NOT_INITIALIZED}
	ErrDeinitialized              = &Error{Code: cudasys.CUDA_ERROR_DEINITIALIZED}
	ErrProfilerDisabled           = &Error{Code: cudasys.CUDA_ERROR_PROFILER_DISABLED}
	ErrStubLibrary                = &Error{Code: cudasys.CUDA_ERROR_STUB_LIBRARY}
	ErrDeviceUnavailable          = &Error{Code: cudasys.CUDA_ERROR_DEVICE_UNAVAILABLE}
	ErrNoDevice                   = &Error{Code: cudasys.CUDA_ERROR_NO_DEVICE}
	ErrInvalidDevice              = &Error{Code: cudasys.CUDA_ERROR_INVALID_DEVICE}
	ErrDeviceNotLicensed          = &Error{Code: cudasys.CUDA_ERROR_DEVICE_NOT_LICENSED}
	ErrInvalidImage               = &Error{Code: cudasys.CUDA_ERROR_INVALID_IMAGE}
	ErrInvalidContext             = &Error{Code: cudasys.CUDA_ERROR_INVALID_CONTEXT}
	ErrNoBinaryForGPU             = &Error{Code: cudasys.CUDA_ERROR_NO_BINARY_FOR_GPU}
	ErrInvalidPTX                 = &Error{Code: cudasys.CUDA_ERROR_INVALID_PTX}
	ErrUnsupportedPTXVersion      = &Error{Code: cudasys.CUDA_ERROR_UNSUPPORTED_PTX_VERSION}
	ErrInvalidSource              = &Error{Code: cudasys.CUDA_ERROR_INVALID_SOURCE}
	ErrFileNotFound               = &Error{Code: cudasys.CUDA_ERROR_FILE_NOT_FOUND}
	ErrSharedObjectSymbolNotFound = &Error{Code: cudasys.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND}
	ErrSharedObjectInitFailed     = &Error{Code: cudasys.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED}
	ErrOperatingSystem            = &Error{Code: cudasys.CUDA_ERROR_OPERATING_SYSTEM}
	ErrInvalidHandle              = &Error{Code: cudasys.CUDA_ERROR_INVALID_HANDLE}
	ErrIllegalState               = &Error{Code: cudasys.CUDA_ERROR_ILLEGAL_STATE}
	ErrNotFound                   = &Error{Code: cudasys.CUDA_ERROR_NOT_FOUND}
	ErrNotReady                   = &Error{Code: cudasys.CUDA_ERROR_NOT_READY}
	ErrIllegalAddress             = &Error{Code: cudasys.CUDA_ERROR_ILLEGAL_ADDRESS}
	ErrLaunchOutOfResources       = &Error{Code: cudasys.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES}
	ErrLaunchTimeout              = &Error{Code: cudasys.CUDA_ERROR_LAUNCH_TIMEOUT}
	ErrLaunchFailed               = &Error{Code: cudasys.CUDA_ERROR_LAUNCH_FAILED}
	ErrNotPermitted               = &Error{Code: cudasys.CUDA_ERROR_NOT_PERMITTED}
	ErrNotSupported               = &Error{Code: cudasys.CUDA_ERROR_NOT_SUPPORTED}
	ErrSystemNotReady             = &Error{Code: cudasys.CUDA_ERROR_SYSTEM_NOT_READY}
	ErrSystemDriverMismatch       = &Error{Code: cudasys.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH}
	ErrUnknown                    = &Error{Code: cudasys.CUDA_ERROR_UNKNOWN}
)

func check(op string, code cudasys.CUresult) error {
	if code == cudasys.CUDA_SUCCESS {
		return nil
	}
	return &Error{Code: code, Op: op}
}
