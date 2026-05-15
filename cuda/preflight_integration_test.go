//go:build cuda_integration

package cuda

import (
	"errors"
	"fmt"
	"os"
)

// preflight probes the CUDA driver once at TestMain time. When the driver
// loads but cuInit fails with an environment-level error (e.g. WSL without
// GPU passthrough, no devices, system not ready), the rest of the binary
// is skipped before any unit test runs.
//
// This is the integration-tagged build only. Same binary contains both
// integration tests (which need a working driver) and unit tests (which
// use fake drivers). Some broken environments leave libcuda in a state
// where later activity in the process can segfault even though the unit
// tests themselves never touch CUDA. Exiting cleanly here avoids that.
func preflight() bool {
	if err := Init(); err != nil {
		if errors.Is(err, ErrOperatingSystem) ||
			errors.Is(err, ErrSystemNotReady) ||
			errors.Is(err, ErrNoDevice) ||
			errors.Is(err, ErrDeviceUnavailable) {
			fmt.Fprintf(os.Stderr, "cuda: skipping all tests, driver unusable: %v\n", err)
			return false
		}
	}
	return true
}
