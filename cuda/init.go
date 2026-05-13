package cuda

import (
	"fmt"
	"sync"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/eitamring/gocudrv/internal/dynload"
	"github.com/eitamring/gocudrv/internal/platform"
)

var (
	mu     sync.Mutex
	driver *cudasys.Driver
	loadFn = defaultLoad
)

func defaultLoad() (*cudasys.Driver, error) {
	lib, err := dynload.OpenAny(dynload.Default(), platform.LibraryCandidates())
	if err != nil {
		return nil, fmt.Errorf("cuda: load library: %w", err)
	}
	return cudasys.Load(lib)
}

// Init loads the CUDA driver library and calls cuInit(0). Subsequent calls
// are no-ops once the driver is loaded. The library handle is released if
// cuInit fails, so retries do not leak.
func Init() error {
	mu.Lock()
	defer mu.Unlock()
	if driver != nil {
		return nil
	}
	d, err := loadFn()
	if err != nil {
		return err
	}
	if err := cudaresult.Init(d, 0); err != nil {
		_ = d.Close()
		return err
	}
	driver = d
	return nil
}
