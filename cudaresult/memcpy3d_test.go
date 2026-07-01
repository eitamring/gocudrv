package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestMemcpy3D(t *testing.T) {
	if err := Memcpy3D(nil, &cudasys.Memcpy3D{}); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := Memcpy3D(&cudasys.Driver{}, &cudasys.Memcpy3D{}); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.Memcpy3D
	d := &cudasys.Driver{CuMemcpy3D: func(desc *cudasys.Memcpy3D) cudasys.CUresult {
		got = *desc
		return cudasys.CUDA_SUCCESS
	}}
	want := &cudasys.Memcpy3D{SrcMemoryType: cudasys.MemoryTypeHost, DstMemoryType: cudasys.MemoryTypeDevice, WidthInBytes: 64, Height: 8, Depth: 4}
	if err := Memcpy3D(d, want); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got.WidthInBytes != 64 || got.Height != 8 || got.Depth != 4 || got.SrcMemoryType != cudasys.MemoryTypeHost {
		t.Errorf("desc not forwarded: %+v", got)
	}

	dErr := &cudasys.Driver{CuMemcpy3D: func(*cudasys.Memcpy3D) cudasys.CUresult {
		return cudasys.CUDA_ERROR_INVALID_VALUE
	}}
	if err := Memcpy3D(dErr, &cudasys.Memcpy3D{}); !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}

func TestMemcpy3DAsync(t *testing.T) {
	if err := Memcpy3DAsync(nil, &cudasys.Memcpy3D{}, 0x5151); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := Memcpy3DAsync(&cudasys.Driver{}, &cudasys.Memcpy3D{}, 0x5151); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var gotStream cudasys.CUstream
	d := &cudasys.Driver{CuMemcpy3DAsync: func(_ *cudasys.Memcpy3D, s cudasys.CUstream) cudasys.CUresult {
		gotStream = s
		return cudasys.CUDA_SUCCESS
	}}
	if err := Memcpy3DAsync(d, &cudasys.Memcpy3D{}, 0x5151); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if gotStream != 0x5151 {
		t.Errorf("stream = %#x, want 0x5151", gotStream)
	}
}
