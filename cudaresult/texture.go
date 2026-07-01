package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// ArrayCreate creates a CUDA array described by desc. Bound best-effort, so it
// returns ErrSymbolUnavailable on a driver that lacks the symbol.
func ArrayCreate(d *cudasys.Driver, desc *cudasys.CUDA_ARRAY_DESCRIPTOR) (cudasys.CUarray, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuArrayCreate == nil {
		return 0, ErrSymbolUnavailable
	}
	var h cudasys.CUarray
	if err := check("cuArrayCreate_v2", d.CuArrayCreate(&h, desc)); err != nil {
		return 0, err
	}
	return h, nil
}

// ArrayDestroy destroys the CUDA array h.
func ArrayDestroy(d *cudasys.Driver, h cudasys.CUarray) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuArrayDestroy == nil {
		return ErrSymbolUnavailable
	}
	return check("cuArrayDestroy", d.CuArrayDestroy(h))
}

// TexObjectCreate creates a texture object over the resource in resDesc using
// the sampling parameters in texDesc.
func TexObjectCreate(d *cudasys.Driver, resDesc *cudasys.CUDA_RESOURCE_DESC, texDesc *cudasys.CUDA_TEXTURE_DESC) (cudasys.CUtexObject, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuTexObjectCreate == nil {
		return 0, ErrSymbolUnavailable
	}
	var h cudasys.CUtexObject
	if err := check("cuTexObjectCreate", d.CuTexObjectCreate(&h, resDesc, texDesc, nil)); err != nil {
		return 0, err
	}
	return h, nil
}

// TexObjectDestroy destroys the texture object h.
func TexObjectDestroy(d *cudasys.Driver, h cudasys.CUtexObject) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuTexObjectDestroy == nil {
		return ErrSymbolUnavailable
	}
	return check("cuTexObjectDestroy", d.CuTexObjectDestroy(h))
}
