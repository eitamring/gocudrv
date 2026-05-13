//go:build linux

package platform

// LibraryCandidates returns the paths to try when loading the CUDA driver
// library on linux, including the WSL2 path that the Windows NVIDIA driver
// projects into the WSL filesystem.
func LibraryCandidates() []string {
	return []string{
		"libcuda.so.1",
		"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
		"/usr/lib/wsl/lib/libcuda.so.1",
	}
}
