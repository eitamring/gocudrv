package executor

import (
	"context"
	"testing"
)

type nopJob struct{}

func (nopJob) Run() error { return nil }

func BenchmarkRoundTripJob(b *testing.B) {
	e := New()
	b.Cleanup(func() { _ = e.Close() })
	bg := context.Background()
	j := nopJob{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := e.DoJob(bg, j); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRoundTripClosure(b *testing.B) {
	e := New()
	b.Cleanup(func() { _ = e.Close() })
	bg := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := e.DoCtxWait(bg, func() error { return nil }); err != nil {
			b.Fatal(err)
		}
	}
}
