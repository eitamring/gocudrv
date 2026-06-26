package cuda

import (
	"context"
	"math"

	"github.com/eitamring/gocudrv/cudaresult"
)

// FunctionAttribute identifies a queryable or settable kernel attribute. It
// mirrors CUfunction_attribute.
type FunctionAttribute int32

const (
	FuncAttrMaxThreadsPerBlock            FunctionAttribute = 0
	FuncAttrSharedSizeBytes               FunctionAttribute = 1
	FuncAttrConstSizeBytes                FunctionAttribute = 2
	FuncAttrLocalSizeBytes                FunctionAttribute = 3
	FuncAttrNumRegs                       FunctionAttribute = 4
	FuncAttrPTXVersion                    FunctionAttribute = 5
	FuncAttrBinaryVersion                 FunctionAttribute = 6
	FuncAttrCacheModeCA                   FunctionAttribute = 7
	FuncAttrMaxDynamicSharedSizeBytes     FunctionAttribute = 8
	FuncAttrPreferredSharedMemoryCarveout FunctionAttribute = 9
)

// Attribute returns one kernel attribute via cuFuncGetAttribute, such as
// register or shared-memory usage. It returns ErrSymbolUnavailable on a driver
// that does not export cuFuncGetAttribute.
func (f *Function) Attribute(attr FunctionAttribute) (int, error) {
	if f == nil {
		return 0, ErrNilFunction
	}
	if f.module == nil {
		return 0, ErrNilModule
	}
	f.module.opMu.RLock()
	defer f.module.opMu.RUnlock()
	if f.module.closed {
		return 0, ErrModuleClosed
	}
	var v int
	err := f.module.ctx.do(context.Background(), func() error {
		n, e := cudaresult.FuncGetAttribute(f.module.ctx.driver, f.raw, int32(attr))
		if e != nil {
			return e
		}
		v = n
		return nil
	})
	if err != nil {
		return 0, err
	}
	return v, nil
}

// SetAttribute sets one kernel attribute via cuFuncSetAttribute. value must be
// non-negative and fit a C int.
func (f *Function) SetAttribute(attr FunctionAttribute, value int) error {
	if f == nil {
		return ErrNilFunction
	}
	if value < 0 || value > math.MaxInt32 {
		return ErrInvalidLength
	}
	if f.module == nil {
		return ErrNilModule
	}
	f.module.opMu.RLock()
	defer f.module.opMu.RUnlock()
	if f.module.closed {
		return ErrModuleClosed
	}
	return f.module.ctx.do(context.Background(), func() error {
		return cudaresult.FuncSetAttribute(f.module.ctx.driver, f.raw, int32(attr), int32(value))
	})
}

// SetMaxDynamicSharedMemory raises the dynamic shared memory a launch of f may
// request, which a kernel needs to opt into more than the 48 KB default. It is
// shorthand for SetAttribute(FuncAttrMaxDynamicSharedSizeBytes, bytes).
func (f *Function) SetMaxDynamicSharedMemory(bytes int) error {
	return f.SetAttribute(FuncAttrMaxDynamicSharedSizeBytes, bytes)
}
