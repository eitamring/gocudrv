//go:build cuda_integration && linux

package cudasys

import (
	"testing"

	"github.com/eitamring/gocudrv/internal/dynload"
	"github.com/eitamring/gocudrv/internal/platform"
)

func loadRealDriver(tb testing.TB) *Driver {
	tb.Helper()
	lib, err := dynload.OpenAny(dynload.Default(), platform.LibraryCandidates())
	if err != nil {
		tb.Skipf("driver unavailable: %v", err)
	}
	d, err := Load(lib)
	if err != nil {
		tb.Skipf("Load: %v", err)
	}
	if r := d.CuInit(0); r != CUDA_SUCCESS {
		tb.Skipf("cuInit: %v", r)
	}
	return d
}

// TestSyscallNMatchesRegistered compares the SyscallN dispatch against fresh
// purego-registered bindings for representative call shapes, including error
// paths and a negative scalar; the integration suite is the broad ABI oracle.
func TestSyscallNMatchesRegistered(t *testing.T) {
	d := loadRealDriver(t)

	var regGetVersion func(version *int32) CUresult
	var regDeviceGet func(device *CUdevice, ordinal int32) CUresult
	var regGetAttr func(value *int32, attr int32, dev CUdevice) CUresult
	if err := bind(d.lib, &regGetVersion, "cuDriverGetVersion"); err != nil {
		t.Fatalf("bind cuDriverGetVersion: %v", err)
	}
	if err := bind(d.lib, &regDeviceGet, "cuDeviceGet"); err != nil {
		t.Fatalf("bind cuDeviceGet: %v", err)
	}
	if err := bind(d.lib, &regGetAttr, "cuDeviceGetAttribute"); err != nil {
		t.Fatalf("bind cuDeviceGetAttribute: %v", err)
	}

	var vReg, vSys int32
	if r1, r2 := regGetVersion(&vReg), d.CuDriverGetVersion(&vSys); r1 != r2 || vReg != vSys {
		t.Errorf("version: registered (%v, %d) != syscalln (%v, %d)", r1, vReg, r2, vSys)
	}

	var dev CUdevice
	if r := d.CuDeviceGet(&dev, 0); r != CUDA_SUCCESS {
		t.Fatalf("CuDeviceGet: %v", r)
	}
	var aReg, aSys int32
	if r1, r2 := regGetAttr(&aReg, 1, dev), d.CuDeviceGetAttribute(&aSys, 1, dev); r1 != r2 || aReg != aSys {
		t.Errorf("attribute: registered (%v, %d) != syscalln (%v, %d)", r1, aReg, r2, aSys)
	}

	var scratch CUdevice
	if r1, r2 := regDeviceGet(&scratch, 9999), d.CuDeviceGet(&scratch, 9999); r1 != r2 {
		t.Errorf("error path: registered %v != syscalln %v", r1, r2)
	}
	if r1, r2 := regDeviceGet(&scratch, -1), d.CuDeviceGet(&scratch, -1); r1 != r2 {
		t.Errorf("negative ordinal: registered %v != syscalln %v", r1, r2)
	}
}

func BenchmarkRegisteredDeviceAttr(b *testing.B) {
	d := loadRealDriver(b)
	var reg func(value *int32, attr int32, dev CUdevice) CUresult
	if err := bind(d.lib, &reg, "cuDeviceGetAttribute"); err != nil {
		b.Fatalf("bind: %v", err)
	}
	var dev CUdevice
	if r := d.CuDeviceGet(&dev, 0); r != CUDA_SUCCESS {
		b.Fatalf("CuDeviceGet: %v", r)
	}
	var v int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r := reg(&v, 1, dev); r != CUDA_SUCCESS {
			b.Fatal(r)
		}
	}
}

func BenchmarkSyscallNDeviceAttr(b *testing.B) {
	d := loadRealDriver(b)
	var dev CUdevice
	if r := d.CuDeviceGet(&dev, 0); r != CUDA_SUCCESS {
		b.Fatalf("CuDeviceGet: %v", r)
	}
	var v int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r := d.CuDeviceGetAttribute(&v, 1, dev); r != CUDA_SUCCESS {
			b.Fatal(r)
		}
	}
}
