package platform

import "testing"

func TestLibraryCandidatesProperties(t *testing.T) {
	got := LibraryCandidates()
	seen := make(map[string]bool, len(got))
	for i, p := range got {
		if p == "" {
			t.Errorf("[%d] is empty", i)
		}
		if seen[p] {
			t.Errorf("[%d] duplicate %q", i, p)
		}
		seen[p] = true
	}
}
