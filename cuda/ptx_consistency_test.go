package cuda

import (
	"bytes"
	"os"
	"testing"
)

// TestPTXFixtureMatchesExample fails if the example's embedded PTX has
// drifted from the integration test fixture. Both files describe the
// same kernel and must stay byte-identical; the build-ptx.sh script
// regenerates them together. This guard catches the case where someone
// regenerates only one of them or edits one by hand.
func TestPTXFixtureMatchesExample(t *testing.T) {
	fixture, err := os.ReadFile("testdata/vector_add.ptx")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	example, err := os.ReadFile("../examples/vector-add/vector_add.ptx")
	if err != nil {
		t.Fatalf("read example: %v", err)
	}
	if !bytes.Equal(fixture, example) {
		t.Fatalf("PTX drift: cuda/testdata/vector_add.ptx (%d bytes) "+
			"and examples/vector-add/vector_add.ptx (%d bytes) differ. "+
			"Regenerate with `make ptx` or `bash examples/vector-add/build-ptx.sh`.",
			len(fixture), len(example))
	}
}
