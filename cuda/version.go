package cuda

import "github.com/eitamring/gocudrv/cudaresult"

// DriverVersion returns the installed CUDA driver version. The value follows
// the CUDA convention (12030 means 12.3). Init must succeed before calling.
func DriverVersion() (int, error) {
	mu.Lock()
	defer mu.Unlock()
	if driver == nil {
		return 0, ErrNotInitialized
	}
	return cudaresult.DriverVersion(driver)
}
