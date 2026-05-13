package cudaresult

import "unsafe"

// unsafeSliceFromPtr is a tiny test helper that mirrors the CUDA convention
// of passing a buffer pointer plus length. It lets a fake driver fill the
// caller's buffer without test code having to round-trip through the
// real cuDeviceGetName signature.
func unsafeSliceFromPtr(p *byte, n int) []byte {
	return unsafe.Slice(p, n)
}
