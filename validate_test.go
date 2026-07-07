package jabcode

import (
	"image"
	"testing"
)

// TestEncodeInvalidOptions checks that malformed options return an error instead
// of panicking via table indexing.
func TestEncodeInvalidOptions(t *testing.T) {
	data := []byte("x")
	cases := []struct {
		name string
		enc  *Encoder
	}{
		{"ecc level too high", NewEncoder(WithECCLevel(11))},
		{"ecc level negative", NewEncoder(WithECCLevel(-1))},
		{"invalid colors", NewEncoder(WithColors(7))},
		{"multi-symbol over color limit", NewEncoder(WithColors(64), WithSymbols(
			[]int{0, 2}, []image.Point{{X: 4, Y: 4}, {X: 4, Y: 4}}, []int{0, 0}))},
		{"symbol position out of range", NewEncoder(WithSymbols(
			[]int{0, 61}, []image.Point{{X: 4, Y: 4}, {X: 4, Y: 4}}, []int{0, 0}))},
		{"secondary ecc too high", NewEncoder(WithSymbols(
			[]int{0, 2}, []image.Point{{X: 4, Y: 4}, {X: 4, Y: 4}}, []int{0, 11}))},
		{"mismatched WithSymbols lengths", NewEncoder(WithSymbols(
			[]int{0, 2}, []image.Point{{X: 4, Y: 4}}, []int{0, 0}))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.enc.Encode(data); err == nil {
				t.Errorf("expected an error, got nil")
			}
		})
	}
}
