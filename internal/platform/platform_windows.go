//go:build windows

package platform

// LibraryCandidates returns the paths to try when loading the CUDA driver
// library on windows. nvcuda.dll is resolved via the standard DLL search
// order.
func LibraryCandidates() []string {
	return []string{"nvcuda.dll"}
}
