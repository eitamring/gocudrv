package dynload

import (
	"errors"
	"testing"
)

type stubLib struct{ h uintptr }

func (s *stubLib) Handle() uintptr { return s.h }
func (s *stubLib) Close() error    { return nil }

type stubOpener struct {
	successAt int
	calls     []string
}

func (s *stubOpener) Open(path string) (Library, error) {
	s.calls = append(s.calls, path)
	if len(s.calls)-1 == s.successAt {
		return &stubLib{h: 42}, nil
	}
	return nil, errors.New("nope")
}

func TestOpenAny(t *testing.T) {
	cases := []struct {
		name      string
		paths     []string
		successAt int
		wantPath  string
		wantErr   error
	}{
		{"first ok", []string{"a", "b", "c"}, 0, "a", nil},
		{"middle ok", []string{"a", "b", "c"}, 1, "b", nil},
		{"last ok", []string{"a", "b", "c"}, 2, "c", nil},
		{"empty paths", nil, 0, "", ErrNoCandidates},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			o := &stubOpener{successAt: tc.successAt}
			lib, err := OpenAny(o, tc.paths)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if lib == nil {
				t.Fatal("nil lib")
			}
			if got := o.calls[len(o.calls)-1]; got != tc.wantPath {
				t.Errorf("last path = %q, want %q", got, tc.wantPath)
			}
		})
	}
}

func TestOpenAnyAllFail(t *testing.T) {
	o := &stubOpener{successAt: -1}
	lib, err := OpenAny(o, []string{"a", "b"})
	if lib != nil {
		t.Error("want nil lib")
	}
	if err == nil {
		t.Fatal("want error")
	}
	if got := len(o.calls); got != 2 {
		t.Errorf("tried %d, want 2", got)
	}
}
