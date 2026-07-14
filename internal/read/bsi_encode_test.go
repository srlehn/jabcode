//go:build jabcode_bsi && jabcode_non_iso_encode

package read

import (
	"fmt"
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestBSIMultiSymbolEncodeRoundTrip(t *testing.T) {
	tests := []struct {
		name      string
		positions []int
		versions  []image.Point
		levels    []int
	}{
		{
			name:      "custom side above",
			positions: []int{0, 1},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(3, 4)},
		},
		{
			name:      "custom side below",
			positions: []int{0, 2},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(3, 4)},
		},
		{
			name:      "custom side left",
			positions: []int{0, 3},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(5, 2)},
		},
		{
			name:      "same-size horizontal",
			positions: []int{0, 4},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(3, 2)},
		},
		{
			name:      "custom horizontal side",
			positions: []int{0, 4},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(5, 2)},
		},
		{
			name:      "distinct secondary error correction",
			positions: []int{0, 4},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(5, 2)},
			levels:    []int{5, 9},
		},
		{
			name:      "secondary hosts another secondary",
			positions: []int{0, 4, 7},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(5, 2), image.Pt(5, 4)},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			payload := []byte("BSI multi-symbol encoder " + test.name)
			levels := test.levels
			if levels == nil {
				levels = make([]int, len(test.positions))
			}
			img, err := encode.Run(encode.Config{
				Colors: 8, ModuleSize: 12, Format: wire.EncodeBSI,
				SymbolNumber: len(test.positions), SymbolPositions: test.positions,
				SymbolVersions: test.versions, SymbolECCLevels: levels,
			}, payload)
			if err != nil {
				t.Fatal(err)
			}
			got, err := DecodeOnly(img, wire.BSI)
			if err != nil {
				t.Fatalf("%v (%s)", err, bsiSecondaryStageSummary(img))
			}
			if string(got) != string(payload) {
				t.Fatalf("payload = %q, want %q", got, payload)
			}
		})
	}
}

func TestBSIMultiSymbolColorModesRoundTrip(t *testing.T) {
	for _, colors := range []int{4, 8, 16, 32, 64, 128, 256} {
		t.Run(fmt.Sprintf("%dc", colors), func(t *testing.T) {
			payload := []byte(fmt.Sprintf("BSI %d-color docked encoder", colors))
			img, err := encode.Run(encode.Config{
				Colors: colors, ModuleSize: 12, Format: wire.EncodeBSI, SymbolNumber: 2,
				SymbolPositions: []int{0, 4},
				SymbolVersions:  []image.Point{image.Pt(3, 2), image.Pt(5, 2)},
				SymbolECCLevels: []int{0, 0},
			}, payload)
			if err != nil {
				t.Fatal(err)
			}
			got, err := DecodeOnly(img, wire.BSI)
			if err != nil {
				t.Fatalf("%v (%s)", err, bsiSecondaryStageSummary(img))
			}
			if string(got) != string(payload) {
				t.Fatalf("payload = %q, want %q", got, payload)
			}
		})
	}
}
