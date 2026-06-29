package jabcode

import (
	"strconv"
	"testing"
)

// mustAtoi parses s as an int, failing the test on error. It is shared by the
// golden-vector tests that read whitespace-delimited reference fixtures.
func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("atoi %q: %v", s, err)
	}
	return n
}
