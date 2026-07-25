package detect

import (
	"math/rand/v2"
	"testing"
)

func leadingEqualNaive(s []byte, v byte) int {
	for i := range s {
		if s[i] != v {
			return i
		}
	}
	return len(s)
}

func trailingEqualNaive(s []byte, v byte) int {
	for i := range s {
		if s[len(s)-1-i] != v {
			return i
		}
	}
	return len(s)
}

// TestRunLengthMatchesNaive pins the word-at-a-time run scans against the
// obvious byte loops. It sweeps every length across the word boundary and
// includes arbitrary byte content, not only the two values the channel bitmaps
// happen to carry, because the scans are relied on not to assume that.
func TestRunLengthMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for length := range 40 {
		for trial := range 200 {
			s := make([]byte, length)
			for i := range s {
				switch trial % 3 {
				case 0:
					s[i] = byte(rng.UintN(256))
				case 1:
					s[i] = []byte{0, 255}[rng.UintN(2)]
				default:
					s[i] = 255
				}
			}
			v := byte(0)
			if trial%2 == 0 && length > 0 {
				v = s[0]
			} else if trial%5 == 0 {
				v = 255
			}
			if got, want := leadingEqual(s, v), leadingEqualNaive(s, v); got != want {
				t.Fatalf("leadingEqual(%v, %d) = %d, want %d", s, v, got, want)
			}
			if got, want := trailingEqual(s, v), trailingEqualNaive(s, v); got != want {
				t.Fatalf("trailingEqual(%v, %d) = %d, want %d", s, v, got, want)
			}
		}
	}
}

// TestRunLengthUniform checks the saturated cases directly, where the scan runs
// off the end of the slice rather than finding a boundary.
func TestRunLengthUniform(t *testing.T) {
	for _, length := range []int{0, 1, 7, 8, 9, 16, 33} {
		s := make([]byte, length)
		for i := range s {
			s[i] = 255
		}
		if got := leadingEqual(s, 255); got != length {
			t.Fatalf("leadingEqual all-255 len %d = %d", length, got)
		}
		if got := trailingEqual(s, 255); got != length {
			t.Fatalf("trailingEqual all-255 len %d = %d", length, got)
		}
		if length > 0 {
			if got := leadingEqual(s, 0); got != 0 {
				t.Fatalf("leadingEqual mismatch len %d = %d", length, got)
			}
			if got := trailingEqual(s, 0); got != 0 {
				t.Fatalf("trailingEqual mismatch len %d = %d", length, got)
			}
		}
	}
}
