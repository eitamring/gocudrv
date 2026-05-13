package cudaresult

import (
	"errors"
	"fmt"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestErrorString(t *testing.T) {
	cases := []struct {
		name string
		err  *Error
		want string
	}{
		{"known no op", &Error{Code: cudasys.CUDA_ERROR_OUT_OF_MEMORY}, "CUDA_ERROR_OUT_OF_MEMORY"},
		{"known with op", &Error{Code: cudasys.CUDA_ERROR_INVALID_VALUE, Op: "cuInit"}, "cuInit: CUDA_ERROR_INVALID_VALUE"},
		{"unknown numeric", &Error{Code: 7777, Op: "cuFoo"}, "cuFoo: CUDA_ERROR_7777"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.err.Error(); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

type wrappedErr struct{ inner error }

func (w *wrappedErr) Error() string { return "wrapped: " + w.inner.Error() }
func (w *wrappedErr) Unwrap() error { return w.inner }

func TestErrorIs(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		target error
		want   bool
	}{
		{"same code different op", &Error{Code: cudasys.CUDA_ERROR_OUT_OF_MEMORY, Op: "x"}, ErrOutOfMemory, true},
		{"different code", &Error{Code: cudasys.CUDA_ERROR_INVALID_VALUE}, ErrOutOfMemory, false},
		{"non error", errors.New("other"), ErrOutOfMemory, false},
		{"wrapped matches via Unwrap", &wrappedErr{inner: &Error{Code: cudasys.CUDA_ERROR_NO_DEVICE}}, ErrNoDevice, true},
		{"fmt errorf wrap matches", fmt.Errorf("ctx: %w", &Error{Code: cudasys.CUDA_ERROR_INVALID_DEVICE}), ErrInvalidDevice, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := errors.Is(tc.err, tc.target); got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCheck(t *testing.T) {
	cases := []struct {
		name     string
		code     cudasys.CUresult
		op       string
		wantNil  bool
		wantCode cudasys.CUresult
		wantOp   string
	}{
		{"success", cudasys.CUDA_SUCCESS, "cuInit", true, 0, ""},
		{"out of memory", cudasys.CUDA_ERROR_OUT_OF_MEMORY, "cuMemAlloc", false, cudasys.CUDA_ERROR_OUT_OF_MEMORY, "cuMemAlloc"},
		{"unknown numeric", 7777, "cuFoo", false, 7777, "cuFoo"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := check(tc.op, tc.code)
			if tc.wantNil {
				if got != nil {
					t.Fatalf("want nil, got %v", got)
				}
				return
			}
			var e *Error
			if !errors.As(got, &e) {
				t.Fatalf("not an *Error: %T", got)
			}
			if e.Code != tc.wantCode {
				t.Errorf("code = %v, want %v", e.Code, tc.wantCode)
			}
			if e.Op != tc.wantOp {
				t.Errorf("op = %q, want %q", e.Op, tc.wantOp)
			}
		})
	}
}
