package cudaresult

import (
	"errors"
	"testing"

	"github.com/eitamring/gocudrv/cudasys"
)

func TestStreamCreate(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUstream
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamCreate: func(stream *cudasys.CUstream, flags uint32) cudasys.CUresult {
				if flags != 1 {
					t.Errorf("flags = %d, want 1", flags)
				}
				*stream = 0x5151
				return cudasys.CUDA_SUCCESS
			}},
			0x5151,
			nil,
		},
		{
			"out of memory",
			&cudasys.Driver{CuStreamCreate: func(*cudasys.CUstream, uint32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StreamCreate(tc.driver, 1)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("stream = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestStreamCreateWithPriority(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUstream
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamCreateWithPriority: func(stream *cudasys.CUstream, flags uint32, priority int32) cudasys.CUresult {
				if flags != 1 {
					t.Errorf("flags = %d, want 1", flags)
				}
				if priority != -1 {
					t.Errorf("priority = %d, want -1", priority)
				}
				*stream = 0x6161
				return cudasys.CUDA_SUCCESS
			}},
			0x6161,
			nil,
		},
		{
			"invalid value",
			&cudasys.Driver{CuStreamCreateWithPriority: func(*cudasys.CUstream, uint32, int32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_VALUE
			}},
			0,
			ErrInvalidValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StreamCreateWithPriority(tc.driver, 1, -1)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("stream = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestStreamDestroy(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamDestroy: func(stream cudasys.CUstream) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuStreamDestroy: func(cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := StreamDestroy(tc.driver, 0x5151)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestStreamSynchronize(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamSynchronize: func(stream cudasys.CUstream) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuStreamSynchronize: func(cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := StreamSynchronize(tc.driver, 0x5151)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestStreamWaitEvent(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuStreamWaitEvent: func(stream cudasys.CUstream, event cudasys.CUevent, flags uint32) cudasys.CUresult {
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				if event != 0xE7E7 {
					t.Errorf("event = %#x, want 0xE7E7", event)
				}
				if flags != 0 {
					t.Errorf("flags = %d, want 0", flags)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuStreamWaitEvent: func(cudasys.CUstream, cudasys.CUevent, uint32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := StreamWaitEvent(tc.driver, 0x5151, 0xE7E7, 0)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestEventCreate(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    cudasys.CUevent
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuEventCreate: func(event *cudasys.CUevent, flags uint32) cudasys.CUresult {
				if flags != 0 {
					t.Errorf("flags = %d, want 0", flags)
				}
				*event = 0xE7E7
				return cudasys.CUDA_SUCCESS
			}},
			0xE7E7,
			nil,
		},
		{
			"out of memory",
			&cudasys.Driver{CuEventCreate: func(*cudasys.CUevent, uint32) cudasys.CUresult {
				return cudasys.CUDA_ERROR_OUT_OF_MEMORY
			}},
			0,
			ErrOutOfMemory,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := EventCreate(tc.driver, 0)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("event = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestEventDestroy(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuEventDestroy: func(event cudasys.CUevent) cudasys.CUresult {
				if event != 0xE7E7 {
					t.Errorf("event = %#x, want 0xE7E7", event)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuEventDestroy: func(cudasys.CUevent) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := EventDestroy(tc.driver, 0xE7E7)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestEventRecord(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuEventRecord: func(event cudasys.CUevent, stream cudasys.CUstream) cudasys.CUresult {
				if event != 0xE7E7 {
					t.Errorf("event = %#x, want 0xE7E7", event)
				}
				if stream != 0x5151 {
					t.Errorf("stream = %#x, want 0x5151", stream)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuEventRecord: func(cudasys.CUevent, cudasys.CUstream) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := EventRecord(tc.driver, 0xE7E7, 0x5151)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestEventQuery(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"ready",
			&cudasys.Driver{CuEventQuery: func(event cudasys.CUevent) cudasys.CUresult {
				if event != 0xE7E7 {
					t.Errorf("event = %#x, want 0xE7E7", event)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"not ready",
			&cudasys.Driver{CuEventQuery: func(cudasys.CUevent) cudasys.CUresult {
				return cudasys.CUDA_ERROR_NOT_READY
			}},
			ErrNotReady,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := EventQuery(tc.driver, 0xE7E7)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr == ErrNotReady {
				var cudaErr *Error
				if !errors.As(err, &cudaErr) || cudaErr.Op != "cuEventQuery" {
					t.Errorf("err = %#v, want cuEventQuery not-ready error", err)
				}
			}
		})
	}
}

func TestEventSynchronize(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		wantErr error
	}{
		{"nil driver", nil, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuEventSynchronize: func(event cudasys.CUevent) cudasys.CUresult {
				if event != 0xE7E7 {
					t.Errorf("event = %#x, want 0xE7E7", event)
				}
				return cudasys.CUDA_SUCCESS
			}},
			nil,
		},
		{
			"invalid handle",
			&cudasys.Driver{CuEventSynchronize: func(cudasys.CUevent) cudasys.CUresult {
				return cudasys.CUDA_ERROR_INVALID_HANDLE
			}},
			ErrInvalidHandle,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := EventSynchronize(tc.driver, 0xE7E7)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("unexpected err: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestEventElapsedTime(t *testing.T) {
	cases := []struct {
		name    string
		driver  *cudasys.Driver
		want    float32
		wantErr error
	}{
		{"nil driver", nil, 0, ErrNotInitialized},
		{"nil func", &cudasys.Driver{}, 0, ErrNotInitialized},
		{
			"success",
			&cudasys.Driver{CuEventElapsedTime: func(ms *float32, start, end cudasys.CUevent) cudasys.CUresult {
				if start != 0x1111 {
					t.Errorf("start = %#x, want 0x1111", start)
				}
				if end != 0x2222 {
					t.Errorf("end = %#x, want 0x2222", end)
				}
				*ms = 1.25
				return cudasys.CUDA_SUCCESS
			}},
			1.25,
			nil,
		},
		{
			"not ready",
			&cudasys.Driver{CuEventElapsedTime: func(*float32, cudasys.CUevent, cudasys.CUevent) cudasys.CUresult {
				return cudasys.CUDA_ERROR_NOT_READY
			}},
			0,
			ErrNotReady,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := EventElapsedTime(tc.driver, 0x1111, 0x2222)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				if tc.wantErr == ErrNotReady {
					var cudaErr *Error
					if !errors.As(err, &cudaErr) || cudaErr.Op != "cuEventElapsedTime" || err.Error() != "cuEventElapsedTime: CUDA_ERROR_NOT_READY" {
						t.Errorf("err = %#v, want cuEventElapsedTime not-ready error", err)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("ms = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestStreamQuery(t *testing.T) {
	if err := StreamQuery(nil, 0x5151); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil driver = %v, want ErrNotInitialized", err)
	}
	if err := StreamQuery(&cudasys.Driver{}, 0x5151); !errors.Is(err, ErrNotInitialized) {
		t.Errorf("nil func = %v, want ErrNotInitialized", err)
	}
	done := &cudasys.Driver{CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_SUCCESS }}
	if err := StreamQuery(done, 0x5151); err != nil {
		t.Errorf("idle = %v, want nil", err)
	}
	pending := &cudasys.Driver{CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_ERROR_NOT_READY }}
	if err := StreamQuery(pending, 0x5151); !errors.Is(err, ErrNotReady) {
		t.Errorf("pending = %v, want ErrNotReady", err)
	} else {
		var cudaErr *Error
		if !errors.As(err, &cudaErr) || cudaErr.Op != "cuStreamQuery" || err.Error() != "cuStreamQuery: CUDA_ERROR_NOT_READY" {
			t.Errorf("pending = %#v, want cuStreamQuery not-ready error", err)
		}
	}
	failed := &cudasys.Driver{CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_ERROR_INVALID_HANDLE }}
	err := StreamQuery(failed, 0x5151)
	var cudaErr *Error
	if !errors.As(err, &cudaErr) || cudaErr.Op != "cuStreamQuery" {
		t.Errorf("failed query = %#v, want cuStreamQuery error", err)
	}
}

func TestNotReadyErrorsAreImmutable(t *testing.T) {
	stream := &cudasys.Driver{CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult {
		return cudasys.CUDA_ERROR_NOT_READY
	}}
	event := &cudasys.Driver{CuEventQuery: func(cudasys.CUevent) cudasys.CUresult {
		return cudasys.CUDA_ERROR_NOT_READY
	}}
	elapsed := &cudasys.Driver{CuEventElapsedTime: func(*float32, cudasys.CUevent, cudasys.CUevent) cudasys.CUresult {
		return cudasys.CUDA_ERROR_NOT_READY
	}}
	cases := []struct {
		name string
		call func() error
		op   string
		text string
	}{
		{"stream", func() error { return StreamQuery(stream, 1) }, "cuStreamQuery", "cuStreamQuery: CUDA_ERROR_NOT_READY"},
		{"event", func() error { return EventQuery(event, 1) }, "cuEventQuery", "cuEventQuery: CUDA_ERROR_NOT_READY"},
		{"elapsed", func() error {
			_, err := EventElapsedTime(elapsed, 1, 2)
			return err
		}, "cuEventElapsedTime", "cuEventElapsedTime: CUDA_ERROR_NOT_READY"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			first := tc.call()
			var exposed *Error
			if !errors.As(first, &exposed) {
				t.Fatalf("errors.As(%T) failed", first)
			}
			exposed.Code = cudasys.CUDA_SUCCESS
			exposed.Op = "changed"

			second := tc.call()
			if !errors.Is(second, ErrNotReady) {
				t.Fatalf("errors.Is(%v, ErrNotReady) = false", second)
			}
			if second.Error() != tc.text {
				t.Errorf("Error() = %q, want %q", second.Error(), tc.text)
			}
			var fresh *Error
			if !errors.As(second, &fresh) {
				t.Fatalf("second errors.As(%T) failed", second)
			}
			if fresh == exposed {
				t.Fatal("errors.As reused a mutable Error")
			}
			if fresh.Code != cudasys.CUDA_ERROR_NOT_READY || fresh.Op != tc.op {
				t.Errorf("errors.As result = %#v, want not-ready %s error", fresh, tc.op)
			}
		})
	}
}

var (
	queryErrorSink error
	elapsedSink    float32
)

func TestQueryNotReadyAllocs(t *testing.T) {
	stream := &cudasys.Driver{CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_ERROR_NOT_READY }}
	event := &cudasys.Driver{CuEventQuery: func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_ERROR_NOT_READY }}
	elapsed := &cudasys.Driver{CuEventElapsedTime: func(*float32, cudasys.CUevent, cudasys.CUevent) cudasys.CUresult {
		return cudasys.CUDA_ERROR_NOT_READY
	}}
	if allocs := testing.AllocsPerRun(1000, func() { queryErrorSink = StreamQuery(stream, 1) }); allocs != 0 {
		t.Errorf("StreamQuery allocations = %v, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(1000, func() { queryErrorSink = EventQuery(event, 1) }); allocs != 0 {
		t.Errorf("EventQuery allocations = %v, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		elapsedSink, queryErrorSink = EventElapsedTime(elapsed, 1, 2)
	}); allocs != 0 {
		t.Errorf("EventElapsedTime not-ready allocations = %v, want 0", allocs)
	}
}

func TestEventElapsedTimeAllocs(t *testing.T) {
	driver := &cudasys.Driver{CuEventElapsedTime: func(ms *float32, _, _ cudasys.CUevent) cudasys.CUresult {
		*ms = 1.25
		return cudasys.CUDA_SUCCESS
	}}
	if allocs := testing.AllocsPerRun(1000, func() {
		var err error
		elapsedSink, err = EventElapsedTime(driver, 1, 2)
		queryErrorSink = err
	}); allocs != 0 {
		t.Errorf("EventElapsedTime allocations = %v, want 0", allocs)
	}
}

func BenchmarkQueryNotReady(b *testing.B) {
	stream := &cudasys.Driver{CuStreamQuery: func(cudasys.CUstream) cudasys.CUresult { return cudasys.CUDA_ERROR_NOT_READY }}
	event := &cudasys.Driver{CuEventQuery: func(cudasys.CUevent) cudasys.CUresult { return cudasys.CUDA_ERROR_NOT_READY }}
	b.Run("stream", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			queryErrorSink = StreamQuery(stream, 1)
		}
	})
	b.Run("event", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			queryErrorSink = EventQuery(event, 1)
		}
	})
}

func BenchmarkEventElapsedTime(b *testing.B) {
	driver := &cudasys.Driver{CuEventElapsedTime: func(ms *float32, _, _ cudasys.CUevent) cudasys.CUresult {
		*ms = 1.25
		return cudasys.CUDA_SUCCESS
	}}
	b.ReportAllocs()
	for b.Loop() {
		elapsedSink, queryErrorSink = EventElapsedTime(driver, 1, 2)
	}
}
