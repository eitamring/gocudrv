package cudaresult

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestArrayCreate(t *testing.T) {
	desc := &cudasys.CUDA_ARRAY_DESCRIPTOR{Width: 64, Height: 8, Format: cudasys.AdFormatFloat, NumChannels: 1}
	if _, err := ArrayCreate(nil, desc); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := ArrayCreate(&cudasys.Driver{}, desc); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.CUDA_ARRAY_DESCRIPTOR
	d := &cudasys.Driver{CuArrayCreate: func(h *cudasys.CUarray, dsc *cudasys.CUDA_ARRAY_DESCRIPTOR) cudasys.CUresult {
		got = *dsc
		*h = 0xA11A7
		return cudasys.CUDA_SUCCESS
	}}
	h, err := ArrayCreate(d, desc)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if h != 0xA11A7 {
		t.Errorf("handle = %#x, want 0xA11A7", h)
	}
	if got != *desc {
		t.Errorf("desc not forwarded: %+v", got)
	}
	dErr := &cudasys.Driver{CuArrayCreate: func(*cudasys.CUarray, *cudasys.CUDA_ARRAY_DESCRIPTOR) cudasys.CUresult {
		return cudasys.CUDA_ERROR_OUT_OF_MEMORY
	}}
	if _, err := ArrayCreate(dErr, desc); !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("err = %v, want ErrOutOfMemory", err)
	}
}

func TestArrayDestroy(t *testing.T) {
	if err := ArrayDestroy(nil, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := ArrayDestroy(&cudasys.Driver{}, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.CUarray
	d := &cudasys.Driver{CuArrayDestroy: func(h cudasys.CUarray) cudasys.CUresult {
		got = h
		return cudasys.CUDA_SUCCESS
	}}
	if err := ArrayDestroy(d, 0xA11A7); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != 0xA11A7 {
		t.Errorf("handle = %#x, want 0xA11A7", got)
	}
}

func TestTexObjectCreate(t *testing.T) {
	res := &cudasys.CUDA_RESOURCE_DESC{ResType: cudasys.ResourceTypeArray, Handle: 0xA11A7}
	tex := &cudasys.CUDA_TEXTURE_DESC{FilterMode: cudasys.FilterModeLinear}
	if _, err := TexObjectCreate(nil, res, tex); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if _, err := TexObjectCreate(&cudasys.Driver{}, res, tex); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var gotRes cudasys.CUDA_RESOURCE_DESC
	var gotTex cudasys.CUDA_TEXTURE_DESC
	var gotView unsafe.Pointer
	d := &cudasys.Driver{CuTexObjectCreate: func(h *cudasys.CUtexObject, r *cudasys.CUDA_RESOURCE_DESC, x *cudasys.CUDA_TEXTURE_DESC, v unsafe.Pointer) cudasys.CUresult {
		gotRes, gotTex, gotView = *r, *x, v
		*h = 0x7E7E
		return cudasys.CUDA_SUCCESS
	}}
	h, err := TexObjectCreate(d, res, tex)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if h != 0x7E7E {
		t.Errorf("handle = %#x, want 0x7E7E", h)
	}
	if gotRes.Handle != 0xA11A7 || gotTex.FilterMode != cudasys.FilterModeLinear {
		t.Errorf("descs not forwarded: res %+v tex %+v", gotRes, gotTex)
	}
	if gotView != nil {
		t.Errorf("resource-view desc = %p, want nil", gotView)
	}
}

func TestTexObjectDestroy(t *testing.T) {
	if err := TexObjectDestroy(nil, 1); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := TexObjectDestroy(&cudasys.Driver{}, 1); !errors.Is(err, ErrSymbolUnavailable) {
		t.Errorf("nil func = %v, want ErrSymbolUnavailable", err)
	}
	var got cudasys.CUtexObject
	d := &cudasys.Driver{CuTexObjectDestroy: func(h cudasys.CUtexObject) cudasys.CUresult {
		got = h
		return cudasys.CUDA_SUCCESS
	}}
	if err := TexObjectDestroy(d, 0x7E7E); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != 0x7E7E {
		t.Errorf("handle = %#x, want 0x7E7E", got)
	}
}
