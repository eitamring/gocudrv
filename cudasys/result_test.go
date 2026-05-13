package cudasys

import "testing"

func TestCUresultName(t *testing.T) {
	cases := []struct {
		name string
		code CUresult
		want string
	}{
		{"success", CUDA_SUCCESS, "CUDA_SUCCESS"},
		{"invalid value", CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE"},
		{"out of memory", CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY"},
		{"not initialized", CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED"},
		{"no device", CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE"},
		{"invalid context", CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT"},
		{"invalid handle", CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE"},
		{"unknown code returns empty", CUresult(7777), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.code.Name(); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
