package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
)

func TestAlignmentSampleCacheReusesMatchingGeometry(t *testing.T) {
	inputVersion := image.Pt(6, 7)
	inputSide := image.Pt(41, 45)
	resolved := core.DecodedSymbol{SideSize: image.Pt(45, 49)}
	resolved.Meta.SideVersion = image.Pt(7, 8)
	matrix := core.NewBitmap(45, 49, 4)
	trace := &detect.AlignmentTrace{Attempted: true}
	var cache alignmentSampleCache
	cache.add(inputVersion, inputSide, false, &resolved, matrix, trace)

	query := core.DecodedSymbol{SideSize: inputSide}
	query.Meta.SideVersion = inputVersion
	got := samplePrimaryByAlignment(nil, [3]*core.Bitmap{}, &query, nil, nil, &cache)
	if got != matrix {
		t.Fatal("matching geometry did not reuse the sampled matrix")
	}
	if query.Meta.SideVersion != resolved.Meta.SideVersion || query.SideSize != resolved.SideSize {
		t.Fatalf("resolved geometry = version %v side %v, want version %v side %v",
			query.Meta.SideVersion, query.SideSize, resolved.Meta.SideVersion, resolved.SideSize)
	}
	if trace.ReuseCount != 1 {
		t.Fatalf("alignment reuse count = %d, want one", trace.ReuseCount)
	}
}

func TestCurrentAlignmentKeepsDeferredChannels(t *testing.T) {
	const width, height = 160, 160
	bm := &core.Bitmap{Width: width, Height: height, Channels: 4, Pix: make([]byte, width*height*4)}
	channels := [3]*core.Bitmap{}
	reads := 0
	for i := range channels {
		ch := &core.Bitmap{Width: width, Height: height, Channels: 1}
		ch.SetPixelReader(func(x, y int) byte {
			reads++
			if x < 0 || y < 0 || x >= width || y >= height {
				return 0
			}
			return 0
		})
		channels[i] = ch
	}
	d := &detect.PrimaryDetector{
		BM: bm,
		Ch: channels,
		FPs: []detect.FinderPattern{
			{Typ: 0, Center: core.PointF{X: 20, Y: 20}, ModuleSize: 5, FoundCount: 1},
			{Typ: 1, Center: core.PointF{X: 140, Y: 20}, ModuleSize: 5, FoundCount: 1},
			{Typ: 2, Center: core.PointF{X: 140, Y: 140}, ModuleSize: 5, FoundCount: 1},
			{Typ: 3, Center: core.PointF{X: 20, Y: 140}, ModuleSize: 5, FoundCount: 1},
		},
	}
	symbol := &core.DecodedSymbol{SideSize: image.Pt(41, 41)}
	symbol.Meta.SideVersion = image.Pt(6, 6)
	matrix := core.NewBitmap(41, 41, 3)
	decodePrimaryMatrixTraced(d, matrix, symbol, nil, nil, nil)
	if got := d.ChannelExpansionCount(); got != 0 {
		t.Fatalf("current alignment expanded deferred channels: %d", got)
	}
	if reads == 0 {
		t.Fatal("current alignment did not read deferred channel pixels")
	}
	for channel, ch := range d.Ch {
		if ch.Pix != nil {
			t.Fatalf("current alignment materialized channel %d", channel)
		}
	}
}
