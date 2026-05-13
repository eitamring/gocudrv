package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// Init calls cuInit through d. Returns ErrNotInitialized if d or the bound
// function is nil; otherwise the returned error reflects the CUresult.
func Init(d *cudasys.Driver, flags uint32) error {
	if d == nil || d.CuInit == nil {
		return ErrNotInitialized
	}
	return check("cuInit", d.CuInit(flags))
}
