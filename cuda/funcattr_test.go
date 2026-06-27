package cuda

import (
	"errors"
	"math"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestFunctionSetMaxDynamicSharedMemory(t *testing.T) {
	ctx, _, fn, _ := newLaunchFixture(t)
	var gotFn cudasys.CUfunction
	var gotAttr, gotVal int32
	ctx.driver.CuFuncSetAttribute = func(f cudasys.CUfunction, attrib, value int32) cudasys.CUresult {
		gotFn, gotAttr, gotVal = f, attrib, value
		return cudasys.CUDA_SUCCESS
	}
	if err := fn.SetMaxDynamicSharedMemory(96 * 1024); err != nil {
		t.Fatalf("SetMaxDynamicSharedMemory: %v", err)
	}
	if gotFn != 0xCAFE || gotAttr != int32(FuncAttrMaxDynamicSharedSizeBytes) || gotVal != 96*1024 {
		t.Errorf("driver got fn=%#x attr=%d val=%d, want 0xCAFE, 8, 98304", gotFn, gotAttr, gotVal)
	}
}

func TestFunctionAttribute(t *testing.T) {
	ctx, _, fn, _ := newLaunchFixture(t)
	ctx.driver.CuFuncGetAttribute = func(v *int32, attrib int32, _ cudasys.CUfunction) cudasys.CUresult {
		if attrib != int32(FuncAttrNumRegs) {
			t.Errorf("attrib = %d, want %d", attrib, FuncAttrNumRegs)
		}
		*v = 40
		return cudasys.CUDA_SUCCESS
	}
	n, err := fn.Attribute(FuncAttrNumRegs)
	if err != nil {
		t.Fatalf("Attribute: %v", err)
	}
	if n != 40 {
		t.Errorf("regs = %d, want 40", n)
	}
}

func TestFunctionAttributeRejects(t *testing.T) {
	ctx, mod, fn, _ := newLaunchFixture(t)
	ctx.driver.CuFuncSetAttribute = func(cudasys.CUfunction, int32, int32) cudasys.CUresult {
		t.Error("driver must not be called on rejected input")
		return cudasys.CUDA_SUCCESS
	}

	var nilFn *Function
	if err := nilFn.SetAttribute(FuncAttrCacheModeCA, 1); !errors.Is(err, ErrNilFunction) {
		t.Errorf("nil function = %v, want ErrNilFunction", err)
	}
	if _, err := nilFn.Attribute(FuncAttrNumRegs); !errors.Is(err, ErrNilFunction) {
		t.Errorf("nil function get = %v, want ErrNilFunction", err)
	}
	if err := fn.SetAttribute(FuncAttrMaxDynamicSharedSizeBytes, -1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("negative value = %v, want ErrInvalidLength", err)
	}
	if err := fn.SetAttribute(FuncAttrMaxDynamicSharedSizeBytes, math.MaxInt32+1); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("overflow value = %v, want ErrInvalidLength", err)
	}
	nilModFn := &Function{}
	if err := nilModFn.SetAttribute(FuncAttrCacheModeCA, 1); !errors.Is(err, ErrNilModule) {
		t.Errorf("nil module = %v, want ErrNilModule", err)
	}

	if err := mod.Close(); err != nil {
		t.Fatalf("close module: %v", err)
	}
	if err := fn.SetAttribute(FuncAttrCacheModeCA, 1); !errors.Is(err, ErrModuleClosed) {
		t.Errorf("closed module = %v, want ErrModuleClosed", err)
	}
}

func TestFunctionAttributeSymbolUnavailable(t *testing.T) {
	// The fixture driver binds neither cuFuncSetAttribute nor cuFuncGetAttribute.
	_, _, fn, _ := newLaunchFixture(t)
	if err := fn.SetAttribute(FuncAttrMaxDynamicSharedSizeBytes, 1024); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("set = %v, want ErrSymbolUnavailable", err)
	}
	if _, err := fn.Attribute(FuncAttrNumRegs); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("get = %v, want ErrSymbolUnavailable", err)
	}
}

func TestFunctionAttributePropagatesError(t *testing.T) {
	ctx, _, fn, _ := newLaunchFixture(t)
	ctx.driver.CuFuncSetAttribute = func(cudasys.CUfunction, int32, int32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}
	if err := fn.SetMaxDynamicSharedMemory(1 << 30); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}
