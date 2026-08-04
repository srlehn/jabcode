//go:build !js

package detect

import (
	"encoding/binary"
	"math"
	"testing"
)

// The kernel's emission modes are pinned by the fusion suite, but that suite
// passes flags straight through, so it says nothing about which mode the route
// actually dispatches with. Two words decide that here and nothing else in the
// build reads them: drop the flag and the device applies its off-line walk,
// which confirms on the seek channel while the host chain confirms on the other
// two and, on real captures, rejects candidates the host keeps; change the
// channel and the sweep reads a mask the host chain was never going to confirm
// against. Either is a silent narrowing of the candidate set.
func TestDirectionalScanParamsSelectTheRouteContract(t *testing.T) {
	const width, height, step = 640, 480, 4
	geom := directionalSweepGeometry(width, height, newScanDirection(45), step)
	params := directionalScanParams(width, height, uint32(1)<<currentFamilySeekChannel, geom)
	if len(params) != finderScanParamsBytes {
		t.Fatalf("parameter block is %d bytes, want %d", len(params), finderScanParamsBytes)
	}
	word := func(i int) uint32 { return binary.LittleEndian.Uint32(params[i*4:]) }

	if got := word(13); got != finderScanSkipCrossCheck {
		t.Errorf("flags word is %#x, want FLAG_SKIP_CROSS_CHECK (%#x)", got, finderScanSkipCrossCheck)
	}
	if got, want := word(2), uint32(1)<<currentFamilySeekChannel; got != want {
		t.Errorf("channel mask is %#x, want %#x for the current family's seek channel", got, want)
	}
	// The remaining words carry the sweep the host would otherwise have walked.
	// A device covering different lines is a different scan, not a faster one.
	for _, c := range []struct {
		name string
		got  uint32
		want uint32
	}{
		{"width", word(0), width},
		{"height", word(1), height},
		{"line length", word(3), uint32(geom.lineLength)},
		{"dx", word(4), math.Float32bits(geom.dx)},
		{"dy", word(5), math.Float32bits(geom.dy)},
		{"nx", word(6), math.Float32bits(geom.nx)},
		{"ny", word(7), math.Float32bits(geom.ny)},
		{"qLo", word(8), math.Float32bits(geom.qLo)},
		{"qStep", word(9), math.Float32bits(geom.qStep)},
		{"line count", word(10), uint32(geom.lineCount)},
	} {
		if c.got != c.want {
			t.Errorf("%s word is %#x, want %#x", c.name, c.got, c.want)
		}
	}
}

func TestDirectionalChainParamsCarryTheHostBasis(t *testing.T) {
	const width, height, count = 640, 480, 37
	base := newScanDirection(30)
	geom := directionalSweepGeometry(width, height, base, 4)
	params := directionalChainParams(width, height, count, true, true, geom, base)
	if len(params) != gpuFinderDirectionalChainParamsBytes {
		t.Fatalf("directional chain parameter block is %d bytes, want %d",
			len(params), gpuFinderDirectionalChainParamsBytes)
	}
	word := func(offset int) uint32 { return binary.LittleEndian.Uint32(params[offset:]) }
	scalarAt := func(offset int) float64 {
		return float64(math.Float32frombits(word(offset)))
	}
	if word(0) != width || word(4) != height || word(8) != count ||
		word(12)&1 == 0 || word(12)&gpuFinderChainFlagColorSource == 0 {
		t.Fatalf("directional chain header = %d %d %d %#x, want %d %d %d print",
			word(0), word(4), word(8), word(12), width, height, count)
	}
	for index, want := range []float64{
		float64(geom.dx), float64(geom.dy),
		float64(geom.nx), float64(geom.ny),
		float64(geom.qLo), float64(geom.qStep),
	} {
		if got := scalarAt(gpuFinderChainParamsSize + index*4); got != float64(float32(want)) {
			t.Fatalf("geometry value %d = %v, want %v", index, got, want)
		}
	}
	offset := gpuFinderChainParamsSize + 24
	for index, direction := range []scanDirection{
		base, base.perpendicular(), base.turn(45), base.turn(-45),
	} {
		for component, want := range []float64{direction.dx, direction.dy, direction.pxPerSample} {
			if got := scalarAt(offset + component*4); got != float64(float32(want)) {
				t.Fatalf("direction %d component %d = %v, want %v", index, component, got, want)
			}
		}
		offset += 12
	}
}
