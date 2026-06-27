package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestFuncSetAttribute(t *testing.T) {
	if err := FuncSetAttribute(nil, 0x1, 8, 65536); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := FuncSetAttribute(&cudasys.Driver{}, 0x1, 8, 65536); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var gotFn cudasys.CUfunction
	var gotAttr, gotVal int32
	d := &cudasys.Driver{CuFuncSetAttribute: func(fn cudasys.CUfunction, attrib, value int32) cudasys.CUresult {
		gotFn, gotAttr, gotVal = fn, attrib, value
		return cudasys.CUDA_SUCCESS
	}}
	if err := FuncSetAttribute(d, 0xCAFE, 8, 65536); err != nil {
		t.Fatalf("FuncSetAttribute: %v", err)
	}
	if gotFn != 0xCAFE || gotAttr != 8 || gotVal != 65536 {
		t.Errorf("driver got fn=%#x attr=%d val=%d, want 0xCAFE, 8, 65536", gotFn, gotAttr, gotVal)
	}

	dErr := &cudasys.Driver{CuFuncSetAttribute: func(cudasys.CUfunction, int32, int32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}}
	if err := FuncSetAttribute(dErr, 0x1, 8, 1); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}

func TestFuncGetAttribute(t *testing.T) {
	if _, err := FuncGetAttribute(nil, 0x1, 4); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := FuncGetAttribute(&cudasys.Driver{}, 0x1, 4); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var gotAttr int32
	var gotFn cudasys.CUfunction
	d := &cudasys.Driver{CuFuncGetAttribute: func(v *int32, attrib int32, fn cudasys.CUfunction) cudasys.CUresult {
		gotAttr, gotFn = attrib, fn
		*v = 32
		return cudasys.CUDA_SUCCESS
	}}
	n, err := FuncGetAttribute(d, 0xCAFE, 4)
	if err != nil {
		t.Fatalf("FuncGetAttribute: %v", err)
	}
	if n != 32 || gotAttr != 4 || gotFn != 0xCAFE {
		t.Errorf("got n=%d attr=%d fn=%#x, want 32, 4, 0xCAFE", n, gotAttr, gotFn)
	}

	dErr := &cudasys.Driver{CuFuncGetAttribute: func(*int32, int32, cudasys.CUfunction) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_HANDLE
	}}
	if _, err := FuncGetAttribute(dErr, 0x1, 4); !errors.Is(err, ErrInvalidHandle) {
		t.Errorf("err = %v, want ErrInvalidHandle", err)
	}
}
