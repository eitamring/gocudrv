//go:build windows

package platform

// LibraryCandidates returns the paths to try when loading the CUDA driver
// library on windows. The dynload opener resolves the bare name from
// System32 only, where the NVIDIA driver installs nvcuda.dll.
func LibraryCandidates() []string {
	return []string{"nvcuda.dll"}
}
