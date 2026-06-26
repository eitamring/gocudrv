package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// ModuleLoadData loads a module from a null-terminated PTX or cubin image in
// memory and returns the opaque module handle.
func ModuleLoadData(d *cudasys.Driver, image *byte) (cudasys.CUmodule, error) {
	if d == nil || d.CuModuleLoadData == nil {
		return 0, ErrNotInitialized
	}
	var mod cudasys.CUmodule
	if err := check("cuModuleLoadData", d.CuModuleLoadData(&mod, image)); err != nil {
		return 0, err
	}
	return mod, nil
}

// ModuleLoadDataEx loads a module with JIT options. options and values are the
// parallel CUjit_option / optionValue arrays cuModuleLoadDataEx expects; values
// holds pointers and integers packed as uintptr. The caller owns any buffers the
// values point at and must keep them alive across the call.
func ModuleLoadDataEx(d *cudasys.Driver, image *byte, options []int32, values []uintptr) (cudasys.CUmodule, error) {
	if d == nil || d.CuModuleLoadDataEx == nil {
		return 0, ErrNotInitialized
	}
	if len(values) != len(options) {
		return 0, ErrInvalidValue
	}
	var optPtr *int32
	var valPtr *uintptr
	if len(options) > 0 {
		optPtr = &options[0]
		valPtr = &values[0]
	}
	var mod cudasys.CUmodule
	if err := check("cuModuleLoadDataEx", d.CuModuleLoadDataEx(&mod, image, uint32(len(options)), optPtr, valPtr)); err != nil {
		return 0, err
	}
	return mod, nil
}

// ModuleUnload releases a module previously returned by ModuleLoadData.
func ModuleUnload(d *cudasys.Driver, mod cudasys.CUmodule) error {
	if d == nil || d.CuModuleUnload == nil {
		return ErrNotInitialized
	}
	return check("cuModuleUnload", d.CuModuleUnload(mod))
}

// ModuleGetFunction looks up a kernel by null-terminated name in a loaded
// module and returns the opaque function handle.
func ModuleGetFunction(d *cudasys.Driver, mod cudasys.CUmodule, name *byte) (cudasys.CUfunction, error) {
	if d == nil || d.CuModuleGetFunction == nil {
		return 0, ErrNotInitialized
	}
	var fn cudasys.CUfunction
	if err := check("cuModuleGetFunction", d.CuModuleGetFunction(&fn, mod, name)); err != nil {
		return 0, err
	}
	return fn, nil
}

// ModuleGetGlobal looks up a __device__ or __constant__ global by
// null-terminated name in a loaded module and returns its device pointer and
// byte size.
func ModuleGetGlobal(d *cudasys.Driver, mod cudasys.CUmodule, name *byte) (cudasys.CUdeviceptr, uint64, error) {
	if d == nil || d.CuModuleGetGlobal == nil {
		return 0, 0, ErrNotInitialized
	}
	var ptr cudasys.CUdeviceptr
	var bytes uint64
	if err := check("cuModuleGetGlobal_v2", d.CuModuleGetGlobal(&ptr, &bytes, mod, name)); err != nil {
		return 0, 0, err
	}
	return ptr, bytes, nil
}
