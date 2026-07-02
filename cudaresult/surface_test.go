package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestArray3DCreate(t *testing.T) {
	desc := &cudasys.CUDA_ARRAY3D_DESCRIPTOR{Width: 64, Height: 8, Format: cudasys.AdFormatFloat, NumChannels: 1, Flags: cudasys.ArraySurfaceLoadStore}
	if _, err := Array3DCreate(nil, desc); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := Array3DCreate(&cudasys.Driver{}, desc); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.CUDA_ARRAY3D_DESCRIPTOR
	d := &cudasys.Driver{CuArray3DCreate: func(h *cudasys.CUarray, dsc *cudasys.CUDA_ARRAY3D_DESCRIPTOR) cudasys.CUresult {
		got = *dsc
		*h = 0xA3D
		return cudasys.CUDA_SUCCESS
	}}
	h, err := Array3DCreate(d, desc)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if h != 0xA3D {
		t.Errorf("handle = %#x, want 0xA3D", h)
	}
	if got != *desc {
		t.Errorf("desc not forwarded: %+v", got)
	}
	dErr := &cudasys.Driver{CuArray3DCreate: func(*cudasys.CUarray, *cudasys.CUDA_ARRAY3D_DESCRIPTOR) cudasys.CUresult {
		return cudasys.CUDA_ERROR_OUT_OF_MEMORY
	}}
	if _, err := Array3DCreate(dErr, desc); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestSurfObjectCreate(t *testing.T) {
	res := &cudasys.CUDA_RESOURCE_DESC{ResType: cudasys.ResourceTypeArray, Handle: 0xA3D}
	if _, err := SurfObjectCreate(nil, res); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := SurfObjectCreate(&cudasys.Driver{}, res); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.CUDA_RESOURCE_DESC
	d := &cudasys.Driver{CuSurfObjectCreate: func(h *cudasys.CUsurfObject, r *cudasys.CUDA_RESOURCE_DESC) cudasys.CUresult {
		got = *r
		*h = 0x5F5F
		return cudasys.CUDA_SUCCESS
	}}
	h, err := SurfObjectCreate(d, res)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if h != 0x5F5F {
		t.Errorf("handle = %#x, want 0x5F5F", h)
	}
	if got.ResType != cudasys.ResourceTypeArray || got.Handle != 0xA3D {
		t.Errorf("resource desc not forwarded: %+v", got)
	}
}

func TestSurfObjectDestroy(t *testing.T) {
	if err := SurfObjectDestroy(nil, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := SurfObjectDestroy(&cudasys.Driver{}, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.CUsurfObject
	d := &cudasys.Driver{CuSurfObjectDestroy: func(h cudasys.CUsurfObject) cudasys.CUresult {
		got = h
		return cudasys.CUDA_SUCCESS
	}}
	if err := SurfObjectDestroy(d, 0x5F5F); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != 0x5F5F {
		t.Errorf("handle = %#x, want 0x5F5F", got)
	}
}
