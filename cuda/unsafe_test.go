package cuda

import "unsafe"

// unsafeSlice mirrors cudaresult's test helper for filling caller-provided
// buffers from a fake driver.
func unsafeSlice(p *byte, n int) []byte { return unsafe.Slice(p, n) }
