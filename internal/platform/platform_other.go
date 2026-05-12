//go:build !linux && !windows

package platform

// LibraryCandidates returns nil on platforms where the CUDA driver is not
// supported.
func LibraryCandidates() []string {
	return nil
}
