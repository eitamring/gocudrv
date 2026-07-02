package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// Array3DCreate creates a CUDA array described by the 3D descriptor (Depth 0
// creates a 2D array). Bound best-effort, so it returns ErrSymbolUnavailable on
// a driver that lacks the symbol.
func Array3DCreate(d *cudasys.Driver, desc *cudasys.CUDA_ARRAY3D_DESCRIPTOR) (cudasys.CUarray, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuArray3DCreate == nil {
		return 0, ErrSymbolUnavailable
	}
	var h cudasys.CUarray
	if err := check("cuArray3DCreate_v2", d.CuArray3DCreate(&h, desc)); err != nil {
		return 0, err
	}
	return h, nil
}

// SurfObjectCreate creates a surface object over the resource in resDesc.
func SurfObjectCreate(d *cudasys.Driver, resDesc *cudasys.CUDA_RESOURCE_DESC) (cudasys.CUsurfObject, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuSurfObjectCreate == nil {
		return 0, ErrSymbolUnavailable
	}
	var h cudasys.CUsurfObject
	if err := check("cuSurfObjectCreate", d.CuSurfObjectCreate(&h, resDesc)); err != nil {
		return 0, err
	}
	return h, nil
}

// SurfObjectDestroy destroys the surface object h.
func SurfObjectDestroy(d *cudasys.Driver, h cudasys.CUsurfObject) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuSurfObjectDestroy == nil {
		return ErrSymbolUnavailable
	}
	return check("cuSurfObjectDestroy", d.CuSurfObjectDestroy(h))
}
