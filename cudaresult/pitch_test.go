package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestMemAllocPitch(t *testing.T) {
	if _, _, err := MemAllocPitch(nil, 256, 4, 4); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, _, err := MemAllocPitch(&cudasys.Driver{}, 256, 4, 4); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}

	var gotW, gotH uint64
	var gotESB uint32
	d := &cudasys.Driver{CuMemAllocPitch: func(ptr *cudasys.CUdeviceptr, pitch *uint64, w, h uint64, esb uint32) cudasys.CUresult {
		gotW, gotH, gotESB = w, h, esb
		*ptr = 0xCAFE
		*pitch = 512
		return cudasys.CUDA_SUCCESS
	}}
	ptr, pitch, err := MemAllocPitch(d, 256, 4, 4)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if ptr != 0xCAFE || pitch != 512 {
		t.Errorf("ptr=%#x pitch=%d, want 0xcafe, 512", ptr, pitch)
	}
	if gotW != 256 || gotH != 4 || gotESB != 4 {
		t.Errorf("args = (%d,%d,%d), want (256,4,4)", gotW, gotH, gotESB)
	}

	dErr := &cudasys.Driver{CuMemAllocPitch: func(*cudasys.CUdeviceptr, *uint64, uint64, uint64, uint32) cudasys.CUresult {
		return cudasys.CUDA_ERROR_OUT_OF_MEMORY
	}}
	if _, _, err := MemAllocPitch(dErr, 256, 4, 4); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestMemcpy2D(t *testing.T) {
	if err := Memcpy2D(nil, &cudasys.Memcpy2D{}); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := Memcpy2D(&cudasys.Driver{}, &cudasys.Memcpy2D{}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.Memcpy2D
	d := &cudasys.Driver{CuMemcpy2D: func(desc *cudasys.Memcpy2D) cudasys.CUresult {
		got = *desc
		return cudasys.CUDA_SUCCESS
	}}
	want := &cudasys.Memcpy2D{SrcMemoryType: cudasys.MemoryTypeHost, DstMemoryType: cudasys.MemoryTypeDevice, WidthInBytes: 64, Height: 8}
	if err := Memcpy2D(d, want); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got.WidthInBytes != 64 || got.Height != 8 || got.SrcMemoryType != cudasys.MemoryTypeHost {
		t.Errorf("desc not forwarded: %+v", got)
	}
}

func TestMemcpy2DAsync(t *testing.T) {
	if err := Memcpy2DAsync(nil, &cudasys.Memcpy2D{}, 0x5151); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := Memcpy2DAsync(&cudasys.Driver{}, &cudasys.Memcpy2D{}, 0x5151); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var gotStream cudasys.CUstream
	d := &cudasys.Driver{CuMemcpy2DAsync: func(_ *cudasys.Memcpy2D, s cudasys.CUstream) cudasys.CUresult {
		gotStream = s
		return cudasys.CUDA_SUCCESS
	}}
	if err := Memcpy2DAsync(d, &cudasys.Memcpy2D{}, 0x5151); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if gotStream != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", gotStream)
	}
}
