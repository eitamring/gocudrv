package main

import (
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

		fmt.Printf("\n%d: %s\n", i, name)
		fmt.Printf("  compute capability : %d.%d\n", maj, min)
		fmt.Printf("  total memory       : %d MiB\n", mem/(1<<20))
		fmt.Printf("  multiprocessors    : %d\n", sm)
		fmt.Printf("  warp size          : %d\n", warp)
		fmt.Printf("  core clock         : %d MHz\n", clock/1000)
		fmt.Printf("  memory bus width   : %d bits\n", busWidth)
	}
}

func fail(op string, err error) {
	fmt.Fprintf(os.Stderr, "device-info: %s: %v\n", op, err)
	os.Exit(1)
}
