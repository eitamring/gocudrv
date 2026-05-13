package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// DriverVersion calls cuDriverGetVersion through d. The returned int follows
// the CUDA convention (12030 means CUDA 12.3). Returns ErrNotInitialized if
// d or the bound function is nil.
func DriverVersion(d *cudasys.Driver) (int, error) {
	if d == nil || d.CuDriverGetVersion == nil {
		return 0, ErrNotInitialized
	}
	var v int32
	if err := check("cuDriverGetVersion", d.CuDriverGetVersion(&v)); err != nil {
		return 0, err
	}
	return int(v), nil
}
