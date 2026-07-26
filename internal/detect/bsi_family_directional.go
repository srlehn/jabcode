//go:build jabcode_bsi || jabcode_legacy

package detect

import (
	"github.com/srlehn/jabcode/internal/core"
)

// The BSI-era signature scanned along an arbitrary direction.
//
// Rotation support belongs to the reader, not to one wire family. A BSI finder
// is built the same way the current one is - joined square references rather
// than QR-style concentric rings - so it has the same order-2 symmetry and
// fails an axis-aligned row scan for the same reason once the symbol tilts more
// than about ten degrees. Giving this family the same directional scan is what
// lets the rotation ladder go without BSI losing arbitrary-angle decoding.

// scanDirectionalBSIFamily is scanBSIFamilyRow generalized from a row to a
// line. It seeks in red, as the row walk does.
func (d *PrimaryDetector) scanDirectionalBSIFamily(base scanDirection, step int, state *primaryFamilyScan) {
	d.sweepDirection(d.Ch[0], base, step, state, d.processDirectionalBSIFamilyHit)
}

// processDirectionalBSIFamilyHit is processBSIFamilyHit with the row index and
// row slices replaced by a point and the scan basis.
func (d *PrimaryDetector) processDirectionalBSIFamilyHit(base scanDirection, centre core.PointF, module0 float64, state *primaryFamilyScan) {
	ch := d.Ch
	stats := d.pass().bsiFamily()
	stats.RawHits++
	d.bsiFamilySeedModules = append(d.bsiFamilySeedModules, module0)
	slack := d.ccSlack(module0)

	centres := [3]core.PointF{centre, centre, centre}
	moduleSize := [3]float64{module0}
	if !crossCheckPatternAlong(ch[1], base, module0*2, &centres[1], &moduleSize[1], slack) ||
		!crossCheckPatternAlong(ch[2], base, module0*2, &centres[2], &moduleSize[2], slack) ||
		!checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return
	}

	var colour [3]int
	for c := range 3 {
		x, y := int(centres[c].X), int(centres[c].Y)
		if x < 0 || x >= ch[c].Width || y < 0 || y >= ch[c].Height {
			return
		}
		colour[c] = core.BoolColor(ch[c].Pix[y*ch[c].Width+x] > 0)
	}

	fp := FinderPattern{
		Center: core.PointF{
			X: (centres[0].X + centres[1].X + centres[2].X) / 3,
			Y: (centres[0].Y + centres[1].Y + centres[2].Y) / 3,
		},
		ModuleSize: (moduleSize[0] + moduleSize[1] + moduleSize[2]) / 3,
		FoundCount: 1,
	}
	if !fp.classifyBSIFamily(colour[0], colour[1], colour[2]) ||
		!crossCheckPatternBSIFamilyAlong(ch, &fp, base, d.ccSlack(fp.ModuleSize)) {
		return
	}
	stats.CrossSurvivors[fp.Typ]++
	if state.fps == nil {
		state.fps = make([]FinderPattern, maxFinderPatterns)
	}
	saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
	if state.total >= maxFinderPatterns-1 {
		state.done = true
	}
}

// crossCheckPatternBSIFamilyAlong is crossCheckPatternBSIFamily in the scan
// basis: the BSI signature must hold in all three channels, so every channel
// takes the same perpendicular, along and diagonal walks.
func crossCheckPatternBSIFamilyAlong(ch [3]*core.Bitmap, fp *FinderPattern, base scanDirection, slack int) bool {
	moduleSizeMax := fp.ModuleSize * 2
	var moduleSize [3]float64
	var centres [3]core.PointF
	var direction, diagonal [3]int
	for c := range 3 {
		centres[c] = fp.Center
		if !crossCheckPatternChAlong(ch[c], fp.Typ, base, moduleSizeMax,
			&moduleSize[c], &centres[c], &direction[c], &diagonal[c], slack) {
			return false
		}
	}
	if !checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return false
	}
	fp.ModuleSize = (moduleSize[0] + moduleSize[1] + moduleSize[2]) / 3
	fp.Center = core.PointF{
		X: (centres[0].X + centres[1].X + centres[2].X) / 3,
		Y: (centres[0].Y + centres[1].Y + centres[2].Y) / 3,
	}
	switch {
	case diagonal[0] == 2 || diagonal[1] == 2 || diagonal[2] == 2:
		fp.direction = 2
	case direction[0]+direction[1]+direction[2] > 0:
		fp.direction = 1
	default:
		fp.direction = -1
	}
	return true
}
