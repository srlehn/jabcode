package diag

import "testing"

func TestDiagImageStartSeq(t *testing.T) {
	names := []string{
		"001_balanced.png",
		"099_old.png",
		"1000_previous.png",
		"notes.png",
		"123_nope.jpg",
		"abc_456.png",
	}
	if got, want := diagImageStartSeq(names), 1000; got != want {
		t.Fatalf("diagImageStartSeq = %d, want %d", got, want)
	}
}
