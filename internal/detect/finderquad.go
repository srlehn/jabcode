package detect

import (
	"math"
	"sort"

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
	p3Index := newFinderCandidateIndex(g[3])
	for _, p0 := range g[0] {
		for _, p1 := range g[1] {
			// The final score rejects module-scale mismatches. Apply that
			// necessary condition before entering the deeper candidate loops so
			// weak true corners remain eligible without paying for impossible
			// triples and quads.
			if ratio(p0.ModuleSize, p1.ModuleSize) > quadModuleTol {
				continue
			}
			for _, p2 := range g[2] {
				if ratio(p0.ModuleSize, p2.ModuleSize) > quadModuleTol ||
					ratio(p1.ModuleSize, p2.ModuleSize) > quadModuleTol {
					continue
				}
				top, right := dist(p0.Center, p1.Center), dist(p1.Center, p2.Center)
				if top <= 0 || right <= 0 {
					continue
				}
				minX := max(p2.Center.X-top*quadEdgeTol, p0.Center.X-right*quadEdgeTol)
				maxX := min(p2.Center.X+top*quadEdgeTol, p0.Center.X+right*quadEdgeTol)
				minY := max(p2.Center.Y-top*quadEdgeTol, p0.Center.Y-right*quadEdgeTol)
				maxY := min(p2.Center.Y+top*quadEdgeTol, p0.Center.Y+right*quadEdgeTol)
				for _, p3 := range p3Index.query(minX, minY, maxX, maxY) {
					if ratio(p0.ModuleSize, p3.ModuleSize) > quadModuleTol ||
						ratio(p1.ModuleSize, p3.ModuleSize) > quadModuleTol ||
						ratio(p2.ModuleSize, p3.ModuleSize) > quadModuleTol {
						continue
					}
					d.Stats.Consensus.GeometryTuples++
					score, ok := ScoreFinderQuad(p0, p1, p2, p3)
					d.Stats.Consensus.GeometryScores++
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

type finderQuadCell struct{ x, y int }

type finderCandidateIndex struct {
	cellSize float64
	cells    map[finderQuadCell][]int
	items    []FinderPattern
}

func newFinderCandidateIndex(items []FinderPattern) finderCandidateIndex {
	var sum float64
	count := 0
	for _, item := range items {
		if item.ModuleSize > 0 {
			sum += item.ModuleSize
			count++
		}
	}
	cellSize := 1.0
	if count > 0 {
		cellSize = max(sum/float64(count)*4, 1)
	}
	index := finderCandidateIndex{
		cellSize: cellSize,
		cells:    make(map[finderQuadCell][]int),
		items:    items,
	}
	for i, item := range items {
		key := index.cell(item.Center)
		index.cells[key] = append(index.cells[key], i)
	}
	return index
}

func (index finderCandidateIndex) cell(point core.PointF) finderQuadCell {
	return finderQuadCell{
		x: int(math.Floor(point.X / index.cellSize)),
		y: int(math.Floor(point.Y / index.cellSize)),
	}
}

func (index finderCandidateIndex) query(minX, minY, maxX, maxY float64) []FinderPattern {
	if minX > maxX || minY > maxY {
		return nil
	}
	minCell, maxCell := index.cell(core.PointF{X: minX, Y: minY}), index.cell(core.PointF{X: maxX, Y: maxY})
	ids := make([]int, 0)
	for y := minCell.y; y <= maxCell.y; y++ {
		for x := minCell.x; x <= maxCell.x; x++ {
			ids = append(ids, index.cells[finderQuadCell{x: x, y: y}]...)
		}
	}
	sort.Ints(ids)
	result := make([]FinderPattern, 0, len(ids))
	for _, id := range ids {
		item := index.items[id]
		if item.Center.X >= minX && item.Center.X <= maxX &&
			item.Center.Y >= minY && item.Center.Y <= maxY {
			result = append(result, item)
		}
	}
	return result
}

// SelectFinderQuadByInterpolatedTriple handles the case where one finder type
// has no consistent candidate at all: three types agree on a consistent triple
// while the fourth is genuinely absent or present only as an off-scale spurious
// hit, so no full four-candidate quad is consistent and SelectFinderQuadByGeometry
// finds nothing. It searches for the best-scoring consistent triple, interpolates
// the missing corner from it with the same geometry finishCurrentFamilyScan uses
// for a single missing finder, and returns the completed quad when it passes the
// ScoreFinderQuad gates. Like the full consensus it only runs after the per-type
// selection is already rejected as inconsistent, so a clean selection is never
// disturbed; a wrong grid it might still assemble is caught downstream by the
// palette-coherence admission gate. It reads mask pixels for the interpolation
// seek, so it materializes them first.
func (d *PrimaryDetector) SelectFinderQuadByInterpolatedTriple() ([4]FinderPattern, bool) {
	if !d.ensureBitmap() || !d.ensureChannels() {
		return [4]FinderPattern{}, false
	}
	var g [4][]FinderPattern
	for _, c := range d.Candidates {
		if c.Typ >= 0 && c.Typ < 4 {
			g[c.Typ] = append(g[c.Typ], c)
		}
	}

	var best [4]FinderPattern
	bestScore := math.Inf(1)
	found := false
	for miss := range 4 {
		var present [3][]FinderPattern
		var ptype [3]int
		k, combos := 0, 1
		for t := range 4 {
			if t == miss {
				continue
			}
			if len(g[t]) == 0 {
				k = -1
				break
			}
			present[k], ptype[k] = g[t], t
			combos *= len(g[t])
			k++
		}
		if k != 3 || combos > maxQuadCombos {
			continue
		}
		for _, a := range present[0] {
			for _, b := range present[1] {
				for _, c := range present[2] {
					// Prune before the interpolation seek: the three present
					// corners must already agree on module scale, or the triple
					// is itself a mis-assembly and the interpolated corner would
					// inherit its distortion.
					msMin := min(a.ModuleSize, b.ModuleSize, c.ModuleSize)
					msMax := max(a.ModuleSize, b.ModuleSize, c.ModuleSize)
					if msMin <= 0 || msMax/msMin > quadModuleTol {
						continue
					}
					var fps [4]FinderPattern
					fps[ptype[0]], fps[ptype[1]], fps[ptype[2]] = a, b, c
					fps[miss] = FinderPattern{Typ: miss}
					d.Stats.Consensus.InterpolatedTriples++
					missing, ok := interpolateMissingPattern(fps[:])
					if !ok || fps[missing].Center.X < 0 ||
						fps[missing].Center.X > float64(d.Ch[0].Width-1) ||
						fps[missing].Center.Y < 0 ||
						fps[missing].Center.Y > float64(d.Ch[0].Height-1) {
						continue
					}
					score, ok := ScoreFinderQuad(fps[0], fps[1], fps[2], fps[3])
					if !ok || score >= bestScore {
						continue
					}
					d.Stats.Consensus.InterpolatedSeeks++
					seekMissingFinderPattern(d.BM, fps[:], missing)
					score, ok = ScoreFinderQuad(fps[0], fps[1], fps[2], fps[3])
					if !ok || score >= bestScore {
						continue
					}
					bestScore, best, found = score, fps, true
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

// accumulateFamilyCandidates merges one binarization pass's finder candidates
// into the family's cross-pass union, deduplicating hits that are the same
// physical finder seen again (same type, centres within a module, comparable
// module size - the saveFinderPattern merge criteria) and keeping the
// better-supported one. The union is what SelectFinderFamily hands the
// consensus search, so a corner found only in an earlier pass stays available
// when a later pass is the one that locates.
func (d *PrimaryDetector) accumulateFamilyCandidates(family FinderFamily, candidates []FinderPattern) {
	if family >= finderFamilyCount {
		return
	}
	dst := d.familyPassCandidates[family]
	for _, c := range candidates {
		merged := false
		for i := range dst {
			e := &dst[i]
			if e.Typ == c.Typ &&
				math.Abs(c.Center.X-e.Center.X) <= c.ModuleSize &&
				math.Abs(c.Center.Y-e.Center.Y) <= c.ModuleSize &&
				(math.Abs(c.ModuleSize-e.ModuleSize) <= e.ModuleSize || math.Abs(c.ModuleSize-e.ModuleSize) <= 1.0) {
				if c.FoundCount > e.FoundCount {
					*e = c
				}
				merged = true
				break
			}
		}
		if !merged {
			dst = append(dst, c)
		}
	}
	d.familyPassCandidates[family] = dst
}

// SelectConsensusQuad assembles a finder quad from the cross-pass candidate
// union when the greedy per-type selection located nothing on any pass: the
// full geometric consensus first, then the consistent-triple interpolation. On
// success it installs the quad as the current-family finder list so the sampler
// can proceed; a wrong grid it assembles is caught downstream by the admission
// gate. Reports whether a quad was installed.
func (d *PrimaryDetector) SelectConsensusQuad() bool {
	d.SelectFinderFamily(FinderFamilyCurrent)
	if len(d.FPs) < 4 {
		return false
	}
	quad, ok := d.SelectFinderQuadByGeometry()
	if !ok {
		quad, ok = d.SelectFinderQuadByInterpolatedTriple()
	}
	if !ok {
		return false
	}
	copy(d.FPs[:4], quad[:])
	return true
}

// Gross-inconsistency thresholds for ConsistentFinderQuad, the gate that
// redirects a per-type finder selection to the geometric consensus search.
// They are deliberately looser than the ScoreFinderQuad acceptance gates: this
// gate only rejects a selection that is clearly a mis-assembly - the observed
// field class A signature is a spurious cluster winning one type so the corners
// disagree on module scale by about a factor of two - while an off-axis but
// genuine quad (foreshortened opposite edges, a perspective module-size
// gradient) must pass so its good decode is never disturbed. The consensus
// search this gate triggers then applies the stricter acceptance gates to any
// replacement, so a bad selection is only ever traded for a demonstrably
// consistent one.
const (
	quadRejectModuleTol = 1.8
	quadRejectEdgeTol   = 1.8
)

// ConsistentFinderQuad reports whether the four selected finder patterns (in
// FP0..FP3 cyclic type order) are consistent enough to sample directly: convex,
// with module sizes and opposite-edge lengths agreeing within the
// gross-inconsistency thresholds. The per-type selection scores each type's
// best by foundCount with no cross-type geometry, so it can pick corners that
// pass the side-size arithmetic yet form a degenerate quad that samples off the
// grid; a false result here routes the selection to SelectFinderQuadByGeometry.
func ConsistentFinderQuad(fps []FinderPattern) bool {
	if len(fps) < 4 {
		return false
	}
	p0, p1, p2, p3 := fps[0], fps[1], fps[2], fps[3]
	if !convexQuad(p0.Center, p1.Center, p2.Center, p3.Center) {
		return false
	}
	msMin := min(p0.ModuleSize, p1.ModuleSize, p2.ModuleSize, p3.ModuleSize)
	msMax := max(p0.ModuleSize, p1.ModuleSize, p2.ModuleSize, p3.ModuleSize)
	if msMin <= 0 || msMax/msMin > quadRejectModuleTol {
		return false
	}
	top := dist(p0.Center, p1.Center)
	right := dist(p1.Center, p2.Center)
	bot := dist(p2.Center, p3.Center)
	left := dist(p3.Center, p0.Center)
	return math.Max(ratio(top, bot), ratio(left, right)) <= quadRejectEdgeTol
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
