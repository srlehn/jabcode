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
