package cuda

import (
	"bytes"
	"os"
	"testing"
)

// TestPTXFixtureMatchesExamples fails if checked-in PTX copies drift from the
// integration test fixture. All files describe the same kernel and must stay
// byte-identical; the build-ptx.sh script regenerates them together. This guard
// catches the case where someone regenerates only one copy or edits one by hand.
func TestPTXFixtureMatchesExamples(t *testing.T) {
	fixture, err := os.ReadFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	examples := []string{
		"../examples/vector-add/vector_add.ptx",
		"../examples/event-pipeline/vector_add.ptx",
	}
	for _, path := range examples {
		example, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if !bytes.Equal(fixture, example) {
			t.Fatalf("PTX drift: cuda/testdata/vector_add.ptx (%d bytes) "+
				"and %s (%d bytes) differ. "+
				"Regenerate with `make ptx` or `bash examples/vector-add/build-ptx.sh`.",
				len(fixture), path, len(example))
		}
	}
}
