package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/eitamring/gocudrv/cuda"
)

func main() {
	if err := cuda.Init(); err != nil {
		fail("init", err)
	}

	n, err := cuda.DeviceCount()
	if err != nil {
		fail("device count", err)
	}
	fmt.Printf("CUDA devices: %d\n", n)
	if n == 0 {
		return
	}

	dv, err := cuda.DriverVersion()
	if err == nil {
		fmt.Printf("driver: %d.%d\n", dv/1000, (dv%1000)/10)
	}

	for i := 0; i < n; i++ {
		d, err := cuda.GetDevice(i)
		if err != nil {
			fail(fmt.Sprintf("device %d", i), err)
		}
		name, err := d.Name()
		if err != nil {
			fail(fmt.Sprintf("device %d name", i), err)
		}
		mem, err := d.TotalMemory()
		if err != nil {
			fail(fmt.Sprintf("device %d memory", i), err)
		}
		maj, min, err := d.ComputeCapability()
		if err != nil {
			fail(fmt.Sprintf("device %d compute capability", i), err)
		}
		sm, err := d.Attribute(cuda.DeviceAttributeMultiprocessorCount)
		if err != nil {
			fail(fmt.Sprintf("device %d multiprocessors", i), err)
		}
		warp, err := d.Attribute(cuda.DeviceAttributeWarpSize)
		if err != nil {
			fail(fmt.Sprintf("device %d warp size", i), err)
		}
		clock, err := d.Attribute(cuda.DeviceAttributeClockRate)
		if err != nil {
			fail(fmt.Sprintf("device %d clock rate", i), err)
		}
		busWidth, err := d.Attribute(cuda.DeviceAttributeGlobalMemoryBusWidth)
		if err != nil {
			fail(fmt.Sprintf("device %d memory bus width", i), err)
		}
		l2, err := d.Attribute(cuda.DeviceAttributeL2CacheSize)
		if err != nil {
			fail(fmt.Sprintf("device %d l2 cache", i), err)
		}
		maxTPM, err := d.Attribute(cuda.DeviceAttributeMaxThreadsPerMultiprocessor)
		if err != nil {
			fail(fmt.Sprintf("device %d max threads per sm", i), err)
		}
		asyncEngines, err := d.Attribute(cuda.DeviceAttributeAsyncEngineCount)
		if err != nil {
			fail(fmt.Sprintf("device %d async engines", i), err)
		}

		fmt.Printf("\n%d: %s\n", i, name)
		fmt.Printf("  pci bus id         : %s\n", diag(d.PCIBusID()))
		fmt.Printf("  uuid               : %s\n", diag(d.UUID()))
		fmt.Printf("  compute capability : %d.%d\n", maj, min)
		fmt.Printf("  total memory       : %d MiB\n", mem/(1<<20))
		fmt.Printf("  multiprocessors    : %d\n", sm)
		fmt.Printf("  max threads per sm : %d\n", maxTPM)
		fmt.Printf("  warp size          : %d\n", warp)
		fmt.Printf("  core clock         : %d MHz\n", clock/1000)
		fmt.Printf("  memory bus width   : %d bits\n", busWidth)
		fmt.Printf("  l2 cache           : %d KiB\n", l2/1024)
		fmt.Printf("  async copy engines : %d\n", asyncEngines)
	}
}

// diag renders an optional diagnostic string. A driver too old to export the
// symbol reports ErrSymbolUnavailable, which is shown as "n/a" rather than
// treated as a fatal error.
func diag(s string, err error) string {
	if errors.Is(err, cuda.ErrSymbolUnavailable) {
		return "n/a (driver lacks the symbol)"
	}
	if err != nil {
		return "error: " + err.Error()
	}
	return s
}

func fail(op string, err error) {
	fmt.Fprintf(os.Stderr, "device-info: %s: %v\n", op, err)
	os.Exit(1)
}
