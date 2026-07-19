package encode

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestOpaqueBitLengthBoundaries(t *testing.T) {
	tests := []struct {
		length int
		want   int
	}{
		{1, 7 + 4 + 8},
		{15, 7 + 4 + 15*8},
		{16, 7 + 4 + 13 + 16*8},
		{maxByteRunLength, 7 + 4 + 13 + maxByteRunLength*8},
		{maxByteRunLength + 1, 2*(7+4) + 13 + (maxByteRunLength+1)*8},
		{2 * maxByteRunLength, 2 * (7 + 4 + 13 + maxByteRunLength*8)},
	}
	for _, test := range tests {
		if got := opaqueBitLength(test.length); got != test.want {
			t.Errorf("opaqueBitLength(%d) = %d, want %d", test.length, got, test.want)
		}
		if got := len(encodeOpaqueData(make([]byte, test.length))); got != test.want {
			t.Errorf("len(encodeOpaqueData(%d bytes)) = %d, want %d", test.length, got, test.want)
		}
	}
}

func TestOpaqueCapacityIsMaximal(t *testing.T) {
	for _, version := range []image.Point{image.Pt(1, 1), image.Pt(8, 5), image.Pt(32, 32)} {
		cfg := Config{
			Colors:          8,
			ModuleSize:      1,
			ECCLevel:        spec.DefaultECCLevel,
			Format:          wire.EncodeISO23634,
			Opaque:          true,
			SymbolNumber:    1,
			SymbolPositions: []int{0},
			SymbolVersions:  []image.Point{version},
			SymbolECCLevels: []int{spec.DefaultECCLevel},
		}
		capacity, err := OpaqueCapacity(cfg)
		if err != nil {
			t.Fatalf("OpaqueCapacity(%v): %v", version, err)
		}
		e := encoder{colors: cfg.Colors, eccLevel: cfg.ECCLevel, format: cfg.Format, opaque: true}
		gross := e.symbolCapacity(version, true)
		weights := spec.ECCWeights[cfg.ECCLevel]
		messageBits := netCapacity(gross, weights[0], weights[1]) - 5
		if opaqueBitLength(capacity) > messageBits {
			t.Fatalf("version %v capacity %d needs %d bits, only %d available", version, capacity, opaqueBitLength(capacity), messageBits)
		}
		if opaqueBitLength(capacity+1) <= messageBits {
			t.Fatalf("version %v capacity %d is not maximal", version, capacity)
		}
	}
}
