//go:build cuda_integration

package cuda

import (
	"context"
	"testing"
)

// TestRealCopyFromHostRangeAsync exercises CopyFromHostRangeAsync on real
// hardware in the exact shape goinfer's expert-slot DMA needs: a stacked
// pinned host buffer holding several "experts", one expert's sub-range
// copied into one "slot" of a device buffer whose other slots must stay
// untouched — the check an offset bug (copying the right bytes to the wrong
// place, or the wrong byte count) would fail that a same-size round trip
// would not.
func TestRealCopyFromHostRangeAsync(t *testing.T) {
	ctx, stream := realPinnedAsyncFixture(t)
	const perExpert = 257 // deliberately not a round number
	const numExperts = 4
	const numSlots = 3
	const expertIdx = 2 // mid-range, not 0 and not the last — asymmetric on purpose
	const slotIdx = 1

	hostAll, err := AllocHost[float32](ctx, perExpert*numExperts)
	if err != nil {
		t.Fatalf("AllocHost: %v", err)
	}
	t.Cleanup(func() { _ = hostAll.Close() })
	src := hostAll.Slice()
	for e := range numExperts {
		for i := range perExpert {
			// A value that encodes which expert it came from, so a wrong
			// srcOffset reads a DIFFERENT expert's data, not just noise.
			src[e*perExpert+i] = float32(e)*1000 + float32(i)*0.5
		}
	}

	slots, err := Alloc[float32](ctx, perExpert*numSlots)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	t.Cleanup(func() { _ = slots.Close() })
	bg := context.Background()
	// Zero every slot first so an untouched region reads back as 0, not
	// whatever device memory happened to hold.
	if err := slots.Zero(bg); err != nil {
		t.Fatalf("Zero: %v", err)
	}

	requirePinnedAsync(t, "CopyFromHostRangeAsync",
		slots.CopyFromHostRangeAsync(bg, stream, slotIdx*perExpert, hostAll, expertIdx*perExpert, perExpert))
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, perExpert*numSlots)
	if err := slots.CopyTo(bg, got); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	want := make([]float32, perExpert*numSlots)
	copy(want[slotIdx*perExpert:(slotIdx+1)*perExpert], src[expertIdx*perExpert:(expertIdx+1)*perExpert])
	checkPinnedAsyncValues(t, got, want)
}
