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
