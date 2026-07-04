package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// A side is at most 145 modules (SideSize's upper bound) and finder centers
// sit 3.5 modules inside each edge, so a finder-to-finder walk crosses at
// most 138 modules; a walk still going past that has diverged.
const maxModuleSteps = 145 - 7 + 4

// LocalModuleCount counts the modules between two finder-pattern centers by
// walking their connecting line one module at a time: each step advances by
// the locally interpolated module size and is then re-centered on the most
// homogeneous window along the line. The module-size measurement error is
// corrected at every module instead of accumulating over the whole distance,
// which is what makes the plain distance/module-size estimate miss the side
// version on large and rectangular symbols. It returns -1 when no
// trustworthy count can be produced (degenerate input or a diverged walk).
//
// This is the Local Sampling method of Bugert, Heeger and Berchtold,
// "Version Detection of JAB Codes" (Electronic Imaging 2025, IPAS-229).
func LocalModuleCount(bm *core.Bitmap, fpA, fpB FinderPattern) int {
	if bm == nil || bm.Channels < 3 || fpA.ModuleSize <= 0 || fpB.ModuleSize <= 0 {
		return -1
	}
	ax, ay := fpA.Center.X, fpA.Center.Y
	bx, by := fpB.Center.X, fpB.Center.Y
	total := math.Hypot(bx-ax, by-ay)
	if total <= 0 {
		return -1
	}
	px, py := ax, ay
	steps := 0
	for {
		dx, dy := bx-px, by-py
		remaining := math.Hypot(dx, dy)
		// Interpolate the local module size by the fraction of the A->B line
		// already traversed (a projection, so snapping jitter cannot push it
		// outside [0, 1]).
		t := ((px-ax)*(bx-ax) + (py-ay)*(by-ay)) / (total * total)
		t = math.Min(math.Max(t, 0), 1)
		lms := fpA.ModuleSize*(1-t) + fpB.ModuleSize*t
		if remaining < lms/2 {
			return steps
		}
		if steps >= maxModuleSteps {
			return -1
		}
		ux, uy := dx/remaining, dy/remaining
		px += ux * lms
		py += uy * lms
		px, py = snapToModuleCenter(bm, px, py, ux, uy, lms)
		steps++
	}
}

// snapToModuleCenter shifts a sampling point along the walk direction to the
// most homogeneous window position within a quarter module, so the point can
// never be pulled across a boundary into a neighbouring module. It stays put
// unless a shifted position is strictly better: runs of same-coloured modules
// are homogeneous throughout, and drifting inside them would add error
// instead of correcting it.
func snapToModuleCenter(bm *core.Bitmap, x, y, ux, uy, lms float64) (float64, float64) {
	m := max(int(lms/4+0.5), 1)
	best, ok := windowDeviation(bm, x, y, m)
	if !ok {
		return x, y
	}
	bx, by := x, y
	for o := -m; o <= m; o++ {
		if o == 0 {
			continue
		}
		cx, cy := x+ux*float64(o), y+uy*float64(o)
		if dev, ok := windowDeviation(bm, cx, cy, m); ok && dev < best {
			best, bx, by = dev, cx, cy
		}
	}
	return bx, by
}

// windowDeviation returns the root mean per-channel variance of the
// (2w+1)-square pixel window centered at (cx, cy): near zero inside a single
// module's colour patch, high where the window straddles a module boundary.
// The variance is per channel: pooling the channels would give a red/blue
// boundary the same value distribution as a pure red patch.
func windowDeviation(bm *core.Bitmap, cx, cy float64, w int) (float64, bool) {
	x0 := max(int(cx+0.5)-w, 0)
	x1 := min(int(cx+0.5)+w, bm.Width-1)
	y0 := max(int(cy+0.5)-w, 0)
	y1 := min(int(cy+0.5)+w, bm.Height-1)
	if x0 > x1 || y0 > y1 {
		return 0, false
	}
	stride := bm.Width * bm.Channels
	var sum, sumSq [3]float64
	for y := y0; y <= y1; y++ {
		off := y*stride + x0*bm.Channels
		for x := x0; x <= x1; x++ {
			for c := range 3 {
				v := float64(bm.Pix[off+c])
				sum[c] += v
				sumSq[c] += v * v
			}
			off += bm.Channels
		}
	}
	n := float64((x1 - x0 + 1) * (y1 - y0 + 1))
	var varSum float64
	for c := range 3 {
		mean := sum[c] / n
		varSum += math.Max(sumSq[c]/n-mean*mean, 0)
	}
	return math.Sqrt(varSum / 3), true
}
