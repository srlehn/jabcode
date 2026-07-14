//go:build jabcode_bsi

package detect

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

const (
	bsiCrossAreaWidth         = 14
	bsiMetadataSampleWidth    = 5
	bsiMetadataSampleHeight   = 20
	bsiSecondaryPatternOffset = 7
)

// BSISecondarySeed retains the two near alignment patterns and projected
// directions needed to finish a BSI secondary after its edge metadata reveals
// the customizable side length.
type BSISecondarySeed struct {
	patterns       [4]FinderPattern
	cores          [4][3]byte
	near           [2]int
	far            [2]int
	angles         [2]float64
	sign           float64
	dockedPosition int
}

// PrepareBSISecondary finds the two alignment patterns adjacent to the host
// and samples the canonical metadata strip spanning the docking edge. It does
// not search the image independently of the established host geometry.
func PrepareBSISecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int) (*BSISecondarySeed, *core.Bitmap) {
	if bm == nil || host == nil || secondary == nil || dockedPosition < 0 || dockedPosition > 3 {
		return nil, nil
	}
	for _, channel := range ch {
		if channel == nil {
			return nil, nil
		}
	}

	hp := host.PatternPositions
	distx01, disty01 := hp[1].X-hp[0].X, hp[1].Y-hp[0].Y
	distx32, disty32 := hp[2].X-hp[3].X, hp[2].Y-hp[3].Y
	distx03, disty03 := hp[3].X-hp[0].X, hp[3].Y-hp[0].Y
	distx12, disty12 := hp[2].X-hp[1].X, hp[2].Y-hp[1].Y

	seed := &BSISecondarySeed{dockedPosition: dockedPosition}
	for patternType := range seed.cores {
		seed.cores[patternType] = bsiSecondaryAlignmentCore(host, patternType)
	}
	var hostPatterns [2]int
	var crossSide image.Point
	switch dockedPosition {
	case 3:
		seed.angles = [2]float64{math.Atan2(disty01, distx01), math.Atan2(disty32, distx32)}
		seed.sign = 1
		seed.near, seed.far = [2]int{ap0, ap3}, [2]int{ap1, ap2}
		hostPatterns = [2]int{fp1, fp2}
		crossSide = image.Pt(bsiCrossAreaWidth, host.SideSize.Y)
		secondary.HostPosition = 2
	case 2:
		seed.angles = [2]float64{math.Atan2(disty32, distx32), math.Atan2(disty01, distx01)}
		seed.sign = -1
		seed.near, seed.far = [2]int{ap2, ap1}, [2]int{ap3, ap0}
		hostPatterns = [2]int{fp3, fp0}
		crossSide = image.Pt(bsiCrossAreaWidth, host.SideSize.Y)
		secondary.HostPosition = 3
	case 1:
		seed.angles = [2]float64{math.Atan2(disty12, distx12), math.Atan2(disty03, distx03)}
		seed.sign = 1
		seed.near, seed.far = [2]int{ap1, ap0}, [2]int{ap2, ap3}
		hostPatterns = [2]int{fp2, fp3}
		crossSide = image.Pt(bsiCrossAreaWidth, host.SideSize.X)
		secondary.HostPosition = 0
	case 0:
		seed.angles = [2]float64{math.Atan2(disty03, distx03), math.Atan2(disty12, distx12)}
		seed.sign = -1
		seed.near, seed.far = [2]int{ap3, ap2}, [2]int{ap0, ap1}
		hostPatterns = [2]int{fp0, fp1}
		crossSide = image.Pt(bsiCrossAreaWidth, host.SideSize.X)
		secondary.HostPosition = 1
	}

	for i := range 2 {
		typ := seed.near[i]
		angle := seed.angles[i]
		center := hp[hostPatterns[i]]
		center.X += seed.sign * bsiSecondaryPatternOffset * host.ModuleSize * math.Cos(angle)
		center.Y += seed.sign * bsiSecondaryPatternOffset * host.ModuleSize * math.Sin(angle)
		seed.patterns[typ] = findBSIFamilyAlignmentPattern(
			ch, center.X, center.Y, host.ModuleSize, typ, seed.cores[typ],
		)
		if seed.patterns[typ].FoundCount == 0 {
			return nil, nil
		}
	}

	pt := core.PerspectiveTransform(
		hp[hostPatterns[0]], seed.patterns[seed.near[0]].Center,
		seed.patterns[seed.near[1]].Center, hp[hostPatterns[1]], crossSide,
	)
	areaSource := [4]core.PointF{
		core.Pt(0, 0), core.Pt(bsiMetadataSampleWidth, 0),
		core.Pt(bsiMetadataSampleWidth, bsiMetadataSampleHeight), core.Pt(0, bsiMetadataSampleHeight),
	}
	left := float64(bsiCrossAreaWidth / 2)
	areaTarget := [4]core.PointF{
		pt.Warp(core.Pt(left, 0)), pt.Warp(core.Pt(left+bsiMetadataSampleWidth, 0)),
		pt.Warp(core.Pt(left+bsiMetadataSampleWidth, bsiMetadataSampleHeight)), pt.Warp(core.Pt(left, bsiMetadataSampleHeight)),
	}
	metadata := SampleSymbol(bm, core.QuadToQuad(areaSource, areaTarget), image.Pt(bsiMetadataSampleWidth, bsiMetadataSampleHeight))
	if metadata == nil {
		return nil, nil
	}
	return seed, metadata
}

// FinishBSISecondary locates the two far alignment patterns using the side
// length decoded from the metadata sample, then samples the complete symbol.
func FinishBSISecondary(bm *core.Bitmap, ch [3]*core.Bitmap, secondary *core.DecodedSymbol, seed *BSISecondarySeed) *core.Bitmap {
	if bm == nil || secondary == nil || seed == nil || secondary.SideSize.X < 7 || secondary.SideSize.Y < 7 {
		return nil
	}
	near0 := seed.patterns[seed.near[0]]
	near1 := seed.patterns[seed.near[1]]
	dockedSide := secondary.SideSize.X
	undockedSide := secondary.SideSize.Y
	if seed.dockedPosition == 2 || seed.dockedPosition == 3 {
		dockedSide, undockedSide = secondary.SideSize.Y, secondary.SideSize.X
	}
	secondary.ModuleSize = math.Hypot(near0.Center.X-near1.Center.X, near0.Center.Y-near1.Center.Y) / float64(dockedSide-bsiSecondaryPatternOffset)
	if secondary.ModuleSize <= 0 || math.IsNaN(secondary.ModuleSize) || math.IsInf(secondary.ModuleSize, 0) {
		return nil
	}

	for i := range 2 {
		near := seed.patterns[seed.near[i]]
		typ := seed.far[i]
		distance := float64(undockedSide-bsiSecondaryPatternOffset) * secondary.ModuleSize
		center := core.Pt(
			near.Center.X+seed.sign*distance*math.Cos(seed.angles[i]),
			near.Center.Y+seed.sign*distance*math.Sin(seed.angles[i]),
		)
		seed.patterns[typ] = findBSIFamilyAlignmentPattern(
			ch, center.X, center.Y, secondary.ModuleSize, typ, seed.cores[typ],
		)
	}
	if !completeSecondaryPatternQuad(bm, &seed.patterns, seed.near, seed.far) {
		return nil
	}
	for i := range 4 {
		secondary.PatternPositions[i] = seed.patterns[i].Center
	}
	secondary.ModuleSize = 0
	for i := range 4 {
		secondary.ModuleSize += seed.patterns[i].ModuleSize
	}
	secondary.ModuleSize /= 4
	pt := core.PerspectiveTransform(
		secondary.PatternPositions[0], secondary.PatternPositions[1],
		secondary.PatternPositions[2], secondary.PatternPositions[3], secondary.SideSize,
	)
	return SampleSymbol(bm, pt, secondary.SideSize)
}

func bsiSecondaryAlignmentCore(symbol *core.DecodedSymbol, patternType int) [3]byte {
	var color [3]byte
	if symbol == nil || symbol.Meta.NC < 0 || symbol.Meta.NC > 7 || patternType < 0 || patternType > 3 {
		return color
	}
	colorNumber := 1 << (symbol.Meta.NC + 1)
	offset := patternType * colorNumber * 3
	if offset+3 <= len(symbol.Palette) {
		for channel := range color {
			if symbol.Palette[offset+channel] > 100 {
				color[channel] = 255
			}
		}
	}
	return color
}

func completeSecondaryPatternQuad(bm *core.Bitmap, patterns *[4]FinderPattern, near, far [2]int) bool {
	a, b := near[0], near[1]
	c, d := far[0], far[1]
	if patterns[c].FoundCount == 0 && patterns[d].FoundCount == 0 {
		return false
	}
	if patterns[c].FoundCount == 0 {
		avgBD := (patterns[b].ModuleSize + patterns[d].ModuleSize) / 2
		avgAD := (patterns[a].ModuleSize + patterns[d].ModuleSize) / 2
		if avgBD <= 0 {
			return false
		}
		patterns[c].Center = core.Pt(
			(patterns[d].Center.X-patterns[b].Center.X)/avgBD*avgAD+patterns[a].Center.X,
			(patterns[d].Center.Y-patterns[b].Center.Y)/avgBD*avgAD+patterns[a].Center.Y,
		)
		patterns[c].Typ, patterns[c].FoundCount = c, 1
		patterns[c].ModuleSize = (patterns[a].ModuleSize + patterns[b].ModuleSize + patterns[d].ModuleSize) / 3
	}
	if patterns[d].FoundCount == 0 {
		avgAC := (patterns[a].ModuleSize + patterns[c].ModuleSize) / 2
		avgBC := (patterns[b].ModuleSize + patterns[c].ModuleSize) / 2
		if avgAC <= 0 {
			return false
		}
		patterns[d].Center = core.Pt(
			(patterns[c].Center.X-patterns[a].Center.X)/avgAC*avgBC+patterns[b].Center.X,
			(patterns[c].Center.Y-patterns[a].Center.Y)/avgAC*avgBC+patterns[b].Center.Y,
		)
		patterns[d].Typ, patterns[d].FoundCount = d, 1
		patterns[d].ModuleSize = (patterns[a].ModuleSize + patterns[b].ModuleSize + patterns[c].ModuleSize) / 3
	}
	for _, typ := range far {
		p := patterns[typ].Center
		if p.X < 0 || p.Y < 0 || p.X > float64(bm.Width-1) || p.Y > float64(bm.Height-1) {
			return false
		}
	}
	return true
}
