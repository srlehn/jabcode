package jabcode

import (
	"image"
	"testing"
)

// multiPayload returns n bytes of deterministic mixed-case text so the mode
// encoder switches between upper, lower, numeric and punctuation modes.
func multiPayload(n int) []byte {
	const src = "JAB Code cascades data across docked symbols; 0123456789 mixed CASE text keeps the mode encoder honest. "
	b := make([]byte, n)
	for i := range b {
		b[i] = src[i%len(src)]
	}
	return b
}

// TestEncodeMultiSymbolRoundTrip encodes docked JAB Codes across all four
// docking sides, chained, grid and gapped (non-rectangular) layouts, mixed
// side versions, both colour counts and explicit secondary ECC levels, and
// decodes each back.
//
// Positions index tables.SymbolPos: 0 is the primary at grid {0,0}; 1..4 dock
// a secondary above, below, left and right of it; higher values fill the grid
// outward (6/7/9/10 are the corners of the 3x3 grid, 8 is {0,2}, two below).
// Grid cells left unoccupied render as black modules, so a layout whose
// bounding box is not fully covered exercises the decoder against dead area
// inside the code.
func TestEncodeMultiSymbolRoundTrip(t *testing.T) {
	v4 := image.Pt(4, 4)
	cases := []struct {
		name      string
		colors    int // 0 selects the default (8)
		positions []int
		versions  []image.Point
		eccLevels []int
		payload   int // bytes
	}{
		{"secondary above", 0, []int{0, 1}, []image.Point{v4, v4}, []int{0, 0}, 100},
		{"secondary below", 0, []int{0, 2}, []image.Point{v4, v4}, []int{0, 0}, 100},
		{"secondary left", 0, []int{0, 3}, []image.Point{v4, v4}, []int{0, 0}, 100},
		{"secondary right", 0, []int{0, 4}, []image.Point{v4, v4}, []int{0, 0}, 100},
		{"vertical chain of three", 0, []int{0, 2, 8}, []image.Point{v4, v4, v4}, []int{0, 0, 0}, 150},
		{"chain hosted by a secondary", 0, []int{0, 2, 9}, []image.Point{v4, v4, v4}, []int{0, 0, 0}, 150},
		{"tee of four", 0, []int{0, 1, 3, 4}, []image.Point{v4, v4, v4, v4}, []int{0, 0, 0, 0}, 200},
		{"full cross of five", 0, []int{0, 1, 2, 3, 4}, []image.Point{v4, v4, v4, v4, v4}, []int{0, 0, 0, 0, 0}, 250},
		{"three-by-three grid", 0, []int{0, 1, 2, 3, 4, 6, 7, 9, 10},
			[]image.Point{v4, v4, v4, v4, v4, v4, v4, v4, v4}, []int{0, 0, 0, 0, 0, 0, 0, 0, 0}, 400},
		{"ell of three with a gap", 0, []int{0, 4, 2}, []image.Point{v4, v4, v4}, []int{0, 0, 0}, 140},
		{"ring of eight around a hole", 0, []int{0, 4, 12, 22, 36, 20, 8, 2},
			[]image.Point{v4, v4, v4, v4, v4, v4, v4, v4}, []int{0, 0, 0, 0, 0, 0, 0, 0}, 350},
		{"taller secondary below", 0, []int{0, 2}, []image.Point{v4, image.Pt(4, 6)}, []int{0, 0}, 100},
		{"wider secondary right", 0, []int{0, 4}, []image.Point{v4, image.Pt(6, 4)}, []int{0, 0}, 100},
		{"smaller secondary right", 0, []int{0, 4}, []image.Point{v4, image.Pt(2, 4)}, []int{0, 0}, 60},
		{"four colours below", 4, []int{0, 2}, []image.Point{v4, v4}, []int{0, 0}, 70},
		{"four-colour cross of five", 4, []int{0, 1, 2, 3, 4}, []image.Point{v4, v4, v4, v4, v4}, []int{0, 0, 0, 0, 0}, 160},
		{"sixteen colours below", 16, []int{0, 2}, []image.Point{v4, v4}, []int{0, 0}, 70},
		{"thirty-two colours below", 32, []int{0, 2}, []image.Point{v4, v4}, []int{0, 0}, 70},
		{"sixteen-colour cross of five", 16, []int{0, 1, 2, 3, 4}, []image.Point{v4, v4, v4, v4, v4}, []int{0, 0, 0, 0, 0}, 160},
		{"explicit secondary ecc", 0, []int{0, 2}, []image.Point{v4, v4}, []int{0, 5}, 80},
		{"distinct primary and secondary ecc", 0, []int{0, 2}, []image.Point{v4, v4}, []int{5, 3}, 70},
		{"secondary ecc equal to host", 0, []int{0, 2}, []image.Point{v4, v4}, []int{4, 4}, 80},
		{"mixed version and explicit ecc", 0, []int{0, 2}, []image.Point{v4, image.Pt(4, 6)}, []int{0, 6}, 80},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.colors > 8 && !highColorRoundTripEnabled {
				t.Skip("high-color encoder and decoder not compiled together")
			}
			opts := []Option{WithSymbols(tc.positions, tc.versions, tc.eccLevels)}
			if tc.colors != 0 {
				opts = append(opts, WithColors(tc.colors))
			}
			if tc.colors > 8 {
				opts = append(opts, highColorOption())
			}
			want := multiPayload(tc.payload)
			img, err := NewEncoder(opts...).Encode(want)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			got, err := Decode(img)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			readerTransmission := isoReaderTransmission(want)
			if string(got) != string(readerTransmission) {
				t.Errorf("round-trip mismatch:\n got %q\nwant %q", got, readerTransmission)
			}
		})
	}
}

// TestEncodeMultiSymbolLayoutErrors checks that invalid docked layouts return
// an error instead of producing a broken code.
func TestEncodeMultiSymbolLayoutErrors(t *testing.T) {
	v4 := image.Pt(4, 4)
	cases := []struct {
		name      string
		positions []int
		versions  []image.Point
	}{
		{"duplicate position", []int{0, 2, 2}, []image.Point{v4, v4, v4}},
		{"secondary without host", []int{0, 5}, []image.Point{v4, v4}},
		{"missing primary", []int{1, 2}, []image.Point{v4, v4}},
		{"vertical dock width mismatch", []int{0, 2}, []image.Point{v4, image.Pt(5, 4)}},
		{"horizontal dock height mismatch", []int{0, 4}, []image.Point{v4, image.Pt(4, 5)}},
		{"version out of range", []int{0, 2}, []image.Point{v4, image.Pt(33, 4)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ecc := make([]int, len(tc.positions))
			enc := NewEncoder(WithSymbols(tc.positions, tc.versions, ecc))
			if _, err := enc.Encode(multiPayload(40)); err == nil {
				t.Errorf("expected an error, got nil")
			}
		})
	}
}

// TestEncodeSingleSymbolOptionErrors checks that partially supplied
// WithSymbols calls for a single symbol are rejected instead of silently
// fixing the version while skipping the position and ECC validation.
func TestEncodeSingleSymbolOptionErrors(t *testing.T) {
	v8 := []image.Point{image.Pt(8, 8)}
	cases := []struct {
		name string
		opt  Option
	}{
		{"nil positions with version and ecc", WithSymbols(nil, v8, []int{0})},
		{"nil positions with version only", WithSymbols(nil, v8, nil)},
		{"all nil", WithSymbols(nil, nil, nil)},
		{"empty slices", WithSymbols([]int{}, []image.Point{}, []int{})},
		{"position without version and ecc", WithSymbols([]int{0}, nil, nil)},
		{"non-zero position", WithSymbols([]int{1}, v8, []int{0})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewEncoder(tc.opt).Encode([]byte("probe")); err == nil {
				t.Errorf("expected an error, got nil")
			}
		})
	}
}
