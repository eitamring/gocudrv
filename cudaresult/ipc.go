package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// IpcGetMemHandle exports an IPC handle for a device allocation. Bound
// best-effort, so it returns ErrSymbolUnavailable on a driver that lacks it.
func IpcGetMemHandle(d *cudasys.Driver, dptr cudasys.CUdeviceptr) (cudasys.CUipcMemHandle, error) {
	var h cudasys.CUipcMemHandle
	if d == nil {
		return h, ErrNotInitialized
	}
	if d.CuIpcGetMemHandle == nil {
		return h, ErrSymbolUnavailable
	}
	if err := check("cuIpcGetMemHandle", d.CuIpcGetMemHandle(&h, dptr)); err != nil {
		return cudasys.CUipcMemHandle{}, err
	}
	return h, nil
}

// IpcOpenMemHandle maps another process's exported allocation and returns its
// device pointer in this process.
func IpcOpenMemHandle(d *cudasys.Driver, h cudasys.CUipcMemHandle, flags uint32) (cudasys.CUdeviceptr, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuIpcOpenMemHandle == nil {
		return 0, ErrSymbolUnavailable
	}
	var ptr cudasys.CUdeviceptr
	if err := check("cuIpcOpenMemHandle_v2", d.CuIpcOpenMemHandle(&ptr, h, flags)); err != nil {
		return 0, err
	}
	return ptr, nil
}

// IpcCloseMemHandle unmaps a pointer obtained from IpcOpenMemHandle.
func IpcCloseMemHandle(d *cudasys.Driver, dptr cudasys.CUdeviceptr) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuIpcCloseMemHandle == nil {
		return ErrSymbolUnavailable
	}
	return check("cuIpcCloseMemHandle", d.CuIpcCloseMemHandle(dptr))
}

// IpcGetEventHandle exports an IPC handle for an interprocess event.
func IpcGetEventHandle(d *cudasys.Driver, event cudasys.CUevent) (cudasys.CUipcEventHandle, error) {
	var h cudasys.CUipcEventHandle
	if d == nil {
		return h, ErrNotInitialized
	}
	if d.CuIpcGetEventHandle == nil {
		return h, ErrSymbolUnavailable
	}
	if err := check("cuIpcGetEventHandle", d.CuIpcGetEventHandle(&h, event)); err != nil {
		return cudasys.CUipcEventHandle{}, err
	}
	return h, nil
}

// IpcOpenEventHandle imports another process's exported event.
func IpcOpenEventHandle(d *cudasys.Driver, h cudasys.CUipcEventHandle) (cudasys.CUevent, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuIpcOpenEventHandle == nil {
		return 0, ErrSymbolUnavailable
	}
	var ev cudasys.CUevent
	if err := check("cuIpcOpenEventHandle", d.CuIpcOpenEventHandle(&ev, h)); err != nil {
		return 0, err
	}
	return ev, nil
}
