package cudaresult

import "github.com/eitamring/gocudrv/cudasys"

// FuncSetAttribute sets a CUfunction attribute, such as the maximum dynamic
// shared memory a kernel may request. attrib is a CUfunction_attribute value.
func FuncSetAttribute(d *cudasys.Driver, fn cudasys.CUfunction, attrib, value int32) error {
	if d == nil {
		return ErrNotInitialized
	}
	if d.CuFuncSetAttribute == nil {
		return ErrSymbolUnavailable
	}
	return check("cuFuncSetAttribute", d.CuFuncSetAttribute(fn, attrib, value))
}

// FuncGetAttribute reads a CUfunction attribute such as register or shared
// memory usage.
func FuncGetAttribute(d *cudasys.Driver, fn cudasys.CUfunction, attrib int32) (int, error) {
	if d == nil {
		return 0, ErrNotInitialized
	}
	if d.CuFuncGetAttribute == nil {
		return 0, ErrSymbolUnavailable
	}
	var v int32
	if err := check("cuFuncGetAttribute", d.CuFuncGetAttribute(&v, attrib, fn)); err != nil {
		return 0, err
	}
	return int(v), nil
}
