package cuda

import (
	"errors"
	"os"
	"testing"

	"go.uber.org/goleak"

	"github.com/eitamring/gocudrv/cudasys"
)

// TestMain verifies that no goroutines outlive the test binary. The cuda
// package will own a pinned executor goroutine in later milestones, so the
// check is in place from the start. When the cuda_integration build tag is
// set, the preflight hook also gates against environments where CUDA loads
// but cuInit is unstable (e.g. WSL without GPU passthrough); see
// preflight_integration_test.go.
func TestMain(m *testing.M) {
	if !preflight() {
		os.Exit(0)
	}
	goleak.VerifyTestMain(m)
}

func resetDriver() {
	mu.Lock()
	defer mu.Unlock()
	driver = nil
}

func swapLoad(t *testing.T, fn func() (*cudasys.Driver, error)) {
	t.Helper()
	prev := loadFn
	loadFn = fn
	t.Cleanup(func() { loadFn = prev })
}

func TestDriverVersionBeforeInit(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)
	if _, err := DriverVersion(); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("err = %v, want ErrNotInitialized", err)
	}
}

func TestInit(t *testing.T) {
	loadErr := errors.New("load failed")
	cases := []struct {
		name    string
		load    func() (*cudasys.Driver, error)
		wantErr error
	}{
		{
			"load fails",
			func() (*cudasys.Driver, error) { return nil, loadErr },
			loadErr,
		},
		{
			"cuInit returns out of memory",
			func() (*cudasys.Driver, error) {
				return &cudasys.Driver{
					CuInit: func(uint32) cudasys.CUresult { return cudasys.CUDA_ERROR_OUT_OF_MEMORY },
				}, nil
			},
			ErrOutOfMemory,
		},
		{
			"cuInit returns operating system",
			func() (*cudasys.Driver, error) {
				return &cudasys.Driver{
					CuInit: func(uint32) cudasys.CUresult { return cudasys.CUDA_ERROR_OPERATING_SYSTEM },
				}, nil
			},
			ErrOperatingSystem,
		},
		{
			"success",
			func() (*cudasys.Driver, error) {
				return &cudasys.Driver{
					CuInit: func(uint32) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
				}, nil
			},
			nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resetDriver()
			t.Cleanup(resetDriver)
			swapLoad(t, tc.load)

			err := Init()
			if tc.wantErr == nil {
				if err != nil {
					t.Fatalf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestInitIsIdempotent(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)

	calls := 0
	swapLoad(t, func() (*cudasys.Driver, error) {
		calls++
		return &cudasys.Driver{
			CuInit: func(uint32) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
		}, nil
	})

	for i := 0; i < 3; i++ {
		if err := Init(); err != nil {
			t.Fatalf("init #%d: %v", i, err)
		}
	}
	if calls != 1 {
		t.Errorf("load called %d times, want 1", calls)
	}
}

func TestInitFailureDoesNotRetainDriver(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)
	swapLoad(t, func() (*cudasys.Driver, error) {
		return &cudasys.Driver{
			CuInit: func(uint32) cudasys.CUresult { return cudasys.CUDA_ERROR_OUT_OF_MEMORY },
		}, nil
	})

	if err := Init(); !errors.Is(err, ErrOutOfMemory) {
		t.Fatalf("err = %v, want ErrOutOfMemory", err)
	}

	mu.Lock()
	got := driver
	mu.Unlock()
	if got != nil {
		t.Error("driver retained after failed init")
	}
}

func TestDriverVersionAfterInit(t *testing.T) {
	resetDriver()
	t.Cleanup(resetDriver)
	swapLoad(t, func() (*cudasys.Driver, error) {
		return &cudasys.Driver{
			CuInit:             func(uint32) cudasys.CUresult { return cudasys.CUDA_SUCCESS },
			CuDriverGetVersion: func(v *int32) cudasys.CUresult { *v = 12040; return cudasys.CUDA_SUCCESS },
		}, nil
	})

	if err := Init(); err != nil {
		t.Fatalf("init: %v", err)
	}
	v, err := DriverVersion()
	if err != nil {
		t.Fatalf("driver version: %v", err)
	}
	if v != 12040 {
		t.Errorf("got %d, want 12040", v)
	}
}
