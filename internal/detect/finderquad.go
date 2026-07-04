package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// Geometric gates for the finder-quad consensus fallback. A candidate quad must be
// convex, have opposite edges agreeing within quadEdgeTol (perspective foreshortening
// of an off-axis capture differs the near and far edges), module sizes within
// quadModuleTol, and an edge length per module within quadConsistencyTol of the
// measured module size. The search is exhaustive over candidates — the true corners
// can be low-foundCount on a noisy capture, so candidates are not pruned by
// foundCount — but is skipped if the type groups are large enough that the product of
// combinations would exceed maxQuadCombos, to bound cost on a pathological field.
const (
	maxQuadCombos      = 20_000_000
	quadEdgeTol        = 1.35
	quadModuleTol      = 1.6
	quadConsistencyTol = 1.4
)

// SelectFinderQuadByGeometry searches all finder candidates for the four — one per
// type, in the FP0 FP1 / FP3 FP2 layout — that best form a valid symbol quad. The
// per-type selection in selectBestPatterns scores each type's best by foundCount with
// no cross-type geometry, so on a noisy capture it can pick four candidates that do
// not form a symbol; this consensus search is the fallback. It runs only after the
// normal path fails to yield a valid side size, so clean decodes are untouched.
func (d *PrimaryDetector) SelectFinderQuadByGeometry() ([4]FinderPattern, bool) {
	var g [4][]FinderPattern
	for _, c := range d.Candidates {
		if c.Typ >= 0 && c.Typ < 4 {
			g[c.Typ] = append(g[c.Typ], c)
		}
	}
	combos := 1
	for t := range 4 {
		if len(g[t]) == 0 {
			return [4]FinderPattern{}, false
		}
		combos *= len(g[t])
		if combos > maxQuadCombos {
			return [4]FinderPattern{}, false
		}
	}

	var best [4]FinderPattern
	bestScore := math.Inf(1)
	found := false
	for _, p0 := range g[0] {
		for _, p1 := range g[1] {
			for _, p2 := range g[2] {
				for _, p3 := range g[3] {
					score, ok := ScoreFinderQuad(p0, p1, p2, p3)
					if !ok || score >= bestScore {
						continue
					}
					bestScore = score
					best = [4]FinderPattern{p0, p1, p2, p3}
					found = true
				}
			}
		}
	}
	if !found {
		return best, false
	}
	for i := range best {
		if best[i].direction >= 0 {
			best[i].direction = 1
		} else {
			best[i].direction = -1
		}
	}
	return best, true
}

// ScoreFinderQuad gates and scores a candidate quad (p0,p1,p2,p3 in FP0/FP1/FP2/FP3
// type order, which is cyclic TL,TR,BR,BL around the symbol). It returns a badness
// score (lower is better, 0 = ideal) and whether the quad passes every geometric gate.
func ScoreFinderQuad(p0, p1, p2, p3 FinderPattern) (float64, bool) {
	if !convexQuad(p0.Center, p1.Center, p2.Center, p3.Center) {
		return 0, false
	}
	top := dist(p0.Center, p1.Center)
	right := dist(p1.Center, p2.Center)
	bot := dist(p2.Center, p3.Center)
	left := dist(p3.Center, p0.Center)
	edgeDev := math.Max(ratio(top, bot), ratio(left, right))
	if edgeDev > quadEdgeTol {
		return 0, false
	}
	msMin := min(p0.ModuleSize, p1.ModuleSize, p2.ModuleSize, p3.ModuleSize)
	msMax := max(p0.ModuleSize, p1.ModuleSize, p2.ModuleSize, p3.ModuleSize)
	if msMin <= 0 || msMax/msMin > quadModuleTol {
		return 0, false
	}
	// Geometry-only side size (nil bitmap): this runs inside the exhaustive
	// candidate search, where it is a plausibility gate, not the final answer.
	ss := CalculateSideSize(nil, []FinderPattern{p0, p1, p2, p3})
	if ss.X <= 0 || ss.Y <= 0 {
		return 0, false
	}
	// Edge length per module must match the measured module size, or the quad's
	// geometry and its finders' scales disagree.
	ms := (p0.ModuleSize + p1.ModuleSize + p2.ModuleSize + p3.ModuleSize) / 4
	consist := math.Max(
		math.Max(ratio(top/float64(ss.X), ms), ratio(bot/float64(ss.X), ms)),
		math.Max(ratio(left/float64(ss.Y), ms), ratio(right/float64(ss.Y), ms)),
	)
	if consist > quadConsistencyTol {
		return 0, false
	}
	return (edgeDev - 1) + (msMax/msMin - 1) + (consist - 1), true
}

// convexQuad reports whether p0,p1,p2,p3 listed cyclically form a convex,
// non-self-intersecting quad: all consecutive edge cross-products share one sign.
func convexQuad(p0, p1, p2, p3 core.PointF) bool {
	pts := [4]core.PointF{p0, p1, p2, p3}
	var sign float64
	for i := range 4 {
		a, b, c := pts[i], pts[(i+1)&3], pts[(i+2)&3]
		cross := (b.X-a.X)*(c.Y-b.Y) - (b.Y-a.Y)*(c.X-b.X)
		if cross == 0 {
			return false
		}
		if i == 0 {
			sign = cross
		} else if (cross > 0) != (sign > 0) {
			return false
		}
	}
	return true
}

func dist(a, b core.PointF) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

// ratio returns the larger/smaller ratio of two values (1 = equal), or +Inf if
// either is non-positive.
func ratio(a, b float64) float64 {
	if a <= 0 || b <= 0 {
		return math.Inf(1)
	}
	return math.Max(a, b) / math.Min(a, b)
}
