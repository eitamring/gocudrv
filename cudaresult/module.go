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
