package detect

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// TestShapeOnlyBitmapFailsClosed holds every host stage that indexes balanced
// pixels to the never-panic contract when those pixels are still resident on a
// device. A device-backed detector hands its consumers a bitmap carrying only
// Width, Height and Channels, so a stage that indexes Pix without asking is an
// out-of-range panic reachable from Decode on ordinary input. Each entry point
// has to return its own documented failure instead, which is also what lets the
// route defer the download to the one stage that genuinely needs it.
func TestShapeOnlyBitmapFailsClosed(t *testing.T) {
	shape := &core.Bitmap{Width: 640, Height: 480, Channels: 4}
	if shape.HasPixels() {
		t.Fatal("a bitmap with no Pix reported pixels")
	}
	fps := []FinderPattern{
		{Typ: fp0, Center: core.Pt(20, 20), ModuleSize: 4, FoundCount: 1},
		{Typ: fp1, Center: core.Pt(300, 20), ModuleSize: 4, FoundCount: 1},
		{Typ: fp2, Center: core.Pt(300, 300), ModuleSize: 4, FoundCount: 1},
		{Typ: fp3, Center: core.Pt(20, 300), ModuleSize: 4, FoundCount: 1},
	}
	side := image.Pt(21, 21)
	pt := core.PerspectiveTransform(
		fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, side,
	)

	if got := LocalModuleCount(shape, fps[0], fps[1]); got != -1 {
		t.Errorf("LocalModuleCount = %d, want -1", got)
	}
	// The walk is what needs pixels, so a shape-only bitmap degrades to the
	// finder-distance estimate exactly as a nil one does.
	if got, want := CalculateSideSize(shape, fps), CalculateSideSize(nil, fps); got != want {
		t.Errorf("CalculateSideSize = %v, want the nil-bitmap result %v", got, want)
	}
	if got := averagePixelValue(shape, fps); got != ([3]float32{}) {
		t.Errorf("averagePixelValue = %v, want zero", got)
	}
	if got := SampleSymbol(shape, pt, side); got != nil {
		t.Error("SampleSymbol returned a matrix from a shape-only bitmap")
	}
	if got := SampleSymbolOffset(shape, pt, side, [3]core.PointF{}); got != nil {
		t.Error("SampleSymbolOffset returned a matrix from a shape-only bitmap")
	}
	if got := SearchChannelOffsets(shape, pt, side); got != ([3]core.PointF{}) {
		t.Errorf("SearchChannelOffsets = %v, want zero offsets", got)
	}

	symbol := &core.DecodedSymbol{SideSize: side}
	symbol.Meta.SideVersion = image.Pt(8, 8)
	var trace AlignmentTrace
	ch := [3]*core.Bitmap{
		{Width: shape.Width, Height: shape.Height, Channels: 1},
		{Width: shape.Width, Height: shape.Height, Channels: 1},
		{Width: shape.Width, Height: shape.Height, Channels: 1},
	}
	// The alignment resample now reaches source colour through a block sampler,
	// so the shape-only case is one whose blocks all decline.
	sampler := func(side image.Point, blocks []AlignmentBlock) *core.Bitmap {
		return SampleAlignmentBlocks(shape, side, blocks)
	}
	if got := SampleSymbolByAlignmentPatternTraced(sampler, ch, symbol, fps, &trace); got != nil {
		t.Error("SampleSymbolByAlignmentPatternTraced returned a matrix from a shape-only bitmap")
	}
	if got := SampleSymbolByAlignmentPatternTraced(nil, ch, symbol, fps, &trace); got != nil {
		t.Error("SampleSymbolByAlignmentPatternTraced returned a matrix without a sampler")
	}
	if trace.Reason == "" {
		t.Error("the alignment trace recorded no reason for refusing to sample")
	}
}
