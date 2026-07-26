package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
)

// Directional alignment-pattern scanning: locating a docked secondary's corner
// patterns by turning the scan line rather than the frame.
//
// The walks in detector_ap.go run along image rows, columns and the two image
// diagonals, so they only see a pattern whose module axes are near an image
// axis. That is why a rotated docked symbol needed the frame turned until its
// alignment patterns fell into that band; the pattern was never the problem,
// the basis the walk used was.

// apBasis is the local image-space basis at one predicted alignment-pattern
// position: the images of the symbol's two module axes, plus the two diagonals
// of the module cell they span.
//
// Two vectors rather than one angle, because the module axes are only
// perpendicular in image space when the capture is a pure rotation. Under
// perspective they diverge, and the secondary geometry already measures that
// divergence: findSecondarySymbol derives a separate host-edge angle per
// docking edge because the two are not parallel. A basis built by quarter and
// eighth turns from a single angle would put every walk but the first off the
// pattern by exactly that error, which is why apBasis carries both axes and
// derives its diagonals from them instead of from fixed 45-degree turns.
//
// It is local because it is derived per predicted position. Under perspective
// there is no one basis for the whole symbol; the interior grid's own placement
// arithmetic already re-derives a direction from the nearest located neighbours
// at every cell for the same reason.
type apBasis struct {
	u, v scanDirection

	// dPlus and dMinus are the module cell's diagonals, u+v and u-v. One of
	// them runs along the join between the pattern's two square references and
	// the other crosses the two quadrants that are data; which is which depends
	// on the pattern's diagonal class.
	dPlus, dMinus scanDirection

	// pxPlus and pxMinus convert a run measured in samples along the diagonals
	// into a module width comparable with what the u and v walks report. A cell
	// spanned by a and b has diagonals of length |a+b| and |a-b|, while the
	// axis walks report |a| and |b|, so the factor carries the walk's
	// pxPerSample times the mean axis length over the diagonal's length. For a
	// perpendicular equal-scale basis that is pxPerSample/sqrt(2), reproducing
	// diagPxPerRun exactly.
	pxPlus, pxMinus float64
}

// newAPBasis builds the local basis from two vectors spanning the module axes
// and the module counts they span, so the axes need be neither perpendicular
// nor equally scaled. It reports false for a degenerate basis, where an axis
// has no length or the two are parallel and so span no cell.
//
// The counts are what make this projective rather than merely directional, and
// they are not an optional refinement. Under perspective the two axes generally
// have different pixels-per-module, and the cell's diagonal is then a+b, whose
// direction is not the bisector of the two unit vectors. Normalizing first and
// adding would aim the diagonal walks off the join by that difference, which is
// exactly the error a foreshortened symbol has and an upright or purely rotated
// one does not. Every caller has the counts already: it extrapolated the
// prediction over a known number of modules to get here.
func newAPBasis(ux, uy, uModules, vx, vy, vModules float64) (apBasis, bool) {
	if uModules <= 0 || vModules <= 0 {
		return apBasis{}, false
	}
	// One module along each axis, in pixels.
	ax, ay := ux/uModules, uy/uModules
	bx, by := vx/vModules, vy/vModules

	an, bn := math.Hypot(ax, ay), math.Hypot(bx, by)
	if an == 0 || bn == 0 {
		return apBasis{}, false
	}

	sx, sy := ax+bx, ay+by
	dx, dy := ax-bx, ay-by
	sn, dn := math.Hypot(sx, sy), math.Hypot(dx, dy)
	if sn == 0 || dn == 0 {
		return apBasis{}, false
	}

	b := apBasis{
		u:      scanDirectionFromVector(ax, ay),
		v:      scanDirectionFromVector(bx, by),
		dPlus:  scanDirectionFromVector(sx, sy),
		dMinus: scanDirectionFromVector(dx, dy),
	}
	mean := (an + bn) / 2
	b.pxPlus = b.dPlus.pxPerSample * mean / sn
	b.pxMinus = b.dMinus.pxPerSample * mean / dn
	return b, true
}

// unboundedWalk lets a walk run to the frame edge, for the checks the
// axis-aligned code also bounds only by the image.
const unboundedWalk = 1 << 30

// uprightAPBasis is the basis the axis-aligned walks use implicitly: image x
// and image y, whose diagonals are the image diagonals.
func uprightAPBasis() apBasis {
	b, _ := newAPBasis(1, 0, 1, 0, 1, 1)
	return b
}

// crossCheckAPAlong validates an alignment-pattern candidate along dir and
// refines its centre along that direction.
//
// It is the three-run counterpart of crossCheckAlong rather than a
// generalization of it. An alignment pattern is the finder's construction one
// layer shallower, so a line through its core crosses three runs with the
// middle at index 1, against the finder's five with the middle at index 2. The
// two state machines differ in more than a bound, and parametrizing the run
// count would obscure both to save nothing.
//
// pxPerRun converts a run measured in samples into pixels; it differs from
// dir.pxPerSample on the diagonals, where one module spans the cell diagonal
// rather than its side.
//
// back and fwd bound the walk in samples either side of the seed. The bound is
// what makes the check discriminating rather than nearly always true: a walk
// free to run to the frame edge finds two colour changes somewhere on almost
// any coloured region, so without it a low-resolution symbol accepts nearly
// every core-coloured pixel it seeds from. The axis-aligned walk gets the same
// bound from the search window it is handed.
func crossCheckAPAlong(img *core.Bitmap, dir scanDirection, moduleSizeMax, pxPerRun float64, centre *core.PointF, moduleSize *float64, back, fwd int) bool {
	const stateMiddle = 1
	// The slack the axis-aligned alignment walks use when folding a run back
	// into its neighbour, matching their literal 3.
	const slack = 3
	var sc [3]int

	sx := *centre
	x0, y0 := int(sx.X), int(sx.Y)
	if x0 < 0 || x0 >= img.Width || y0 < 0 || y0 >= img.Height {
		return false
	}

	// Every read goes through Pixel rather than indexing Pix. The alignment
	// path runs against channel bitmaps whose plane may never be materialized:
	// a device-resident mask is read back lazily per pixel, and Pix is then
	// empty. The axis-aligned walks take their samples through an accessor for
	// the same reason.
	at := func(i int, sign float64) (byte, bool) {
		x := int(sx.X + sign*float64(i)*dir.dx)
		y := int(sx.Y + sign*float64(i)*dir.dy)
		if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
			return 0, false
		}
		return img.Pixel(x, y), true
	}

	sc[stateMiddle]++
	prev := img.Pixel(x0, y0)
	si := 0
	for i := 1; si <= stateMiddle && i <= back; i++ {
		cur, ok := at(i, -1)
		if !ok {
			break
		}
		if cur == prev {
			sc[stateMiddle-si]++
		} else if si > 0 && sc[stateMiddle-si] < slack {
			sc[stateMiddle-(si-1)] += sc[stateMiddle-si]
			sc[stateMiddle-si] = 0
			si--
			sc[stateMiddle-si]++
		} else {
			si++
			if si > stateMiddle {
				break
			}
			sc[stateMiddle-si]++
		}
		prev = cur
	}
	if si < stateMiddle {
		return false
	}

	si = 0
	prev = img.Pixel(x0, y0)
	last := 0
	for i := 1; si <= stateMiddle && i <= fwd; i++ {
		cur, ok := at(i, +1)
		if !ok {
			break
		}
		last = i
		if cur == prev {
			sc[stateMiddle+si]++
		} else if si > 0 && sc[stateMiddle+si] < slack {
			sc[stateMiddle+(si-1)] += sc[stateMiddle+si]
			sc[stateMiddle+si] = 0
			si--
			sc[stateMiddle+si]++
		} else {
			si++
			if si > stateMiddle {
				break
			}
			sc[stateMiddle+si]++
		}
		prev = cur
	}
	if si < stateMiddle {
		return false
	}

	ms := float64(sc[1]) * pxPerRun
	if ms >= moduleSizeMax || float64(sc[0]) <= 0.5*float64(sc[1]) || float64(sc[2]) <= 0.5*float64(sc[1]) {
		return false
	}
	*moduleSize = ms

	// A walk moves the centre along its own direction and leaves the
	// perpendicular component alone. Both halves of that matter at low module
	// scale, and getting either wrong costs about a pixel, which is a third of
	// a module at three pixels per module.
	//
	// The measurement is taken from the pixel the walk actually started at,
	// since every sample is read at int(seed + i*step); carrying the seed's
	// fractional part through instead biases the result by up to a pixel. But
	// the correction is then applied to the incoming centre rather than
	// replacing it, so a later walk cannot truncate away what an earlier one
	// refined across it. The axis-aligned walks get this for free by returning
	// one coordinate each.
	offset := float64(last-sc[2]) - float64(sc[1])/2.0
	ux, uy := dir.unit()
	moved := (float64(x0)+offset*dir.dx-sx.X)*ux + (float64(y0)+offset*dir.dy-sx.Y)*uy
	centre.X = sx.X + moved*ux
	centre.Y = sx.Y + moved*uy
	return true
}

// crossCheckAPDiagonalBasis validates a candidate along the module cell's
// diagonals, the directional counterpart of crossCheckPatternDiagonalAP.
//
// Which diagonal is tried first is not a search order. The pattern is two
// squares joined at their corner along one diagonal, so the three-run signature
// exists along that one and not the other; ap0 and ap1 join along u+v and the
// rest along u-v. apx spans both spec types X0 and X1, so its class is a
// property of the hit rather than of the type and the fallback flip is what
// resolves it. dir carries the confirmed diagonal out as +1 or -1.
func crossCheckAPDiagonalBasis(img *core.Bitmap, b apBasis, apType int, moduleSizeMax float64, centre *core.PointF, moduleSize *float64, dir *int) bool {
	plus := apType == ap0 || apType == ap1
	fixed := false
	if *dir != 0 {
		plus, fixed = *dir > 0, true
	}

	for tryCount := 1; ; tryCount++ {
		walk, pxPerRun := b.dMinus, b.pxMinus
		if plus {
			walk, pxPerRun = b.dPlus, b.pxPlus
		}
		probe := *centre
		if crossCheckAPAlong(img, walk, moduleSizeMax, pxPerRun, &probe, moduleSize, unboundedWalk, unboundedWalk) {
			*centre = probe
			if plus {
				*dir = 1
			} else {
				*dir = -1
			}
			return true
		}
		if tryCount == 2 || fixed {
			return false
		}
		plus = !plus
	}
}

// apCoreAt reports whether the pattern's core colour for this channel sits at
// p, the precondition the u walk starts from.
func apCoreAt(img *core.Bitmap, p core.PointF, apType, channel int) bool {
	x, y := int(p.X), int(p.Y)
	if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
		return false
	}
	return img.Pixel(x, y) == palette.Default[apCoreColorIndex(apType)*3+channel]
}

// crossCheckAPUAlong is the u walk with its core-colour precondition, which is
// what the axis-aligned horizontal walk checks before measuring anything.
func crossCheckAPUAlong(img *core.Bitmap, b apBasis, apType, channel int, moduleSizeMax float64, p *core.PointF, ms *float64, back, fwd int) bool {
	return apCoreAt(img, *p, apType, channel) &&
		crossCheckAPAlong(img, b.u, moduleSizeMax, b.u.pxPerSample, p, ms, back, fwd)
}

// crossCheckColorAPBasis runs the core-colour walks over the basis. The core is
// where the two square references overlap, so only one diagonal stays inside
// the pattern and either is accepted, exactly as the axis-aligned check does.
func crossCheckColorAPBasis(img *core.Bitmap, colour int, moduleSize float64, centre core.PointF, b apBasis) bool {
	const (
		moduleNumber = 3
		tol          = 3
	)
	if !crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, b.u, tol) ||
		!crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, b.v, tol) {
		return false
	}
	return crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, b.dPlus, tol) ||
		crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, b.dMinus, tol)
}

// crossCheckPatternAPBasis validates an alignment-pattern candidate across the
// channels and directions of the local basis. It is crossCheckPatternAP with
// the module axes supplied rather than assumed to be the image axes.
//
// The channels are interleaved rather than run one after the other, which is
// load-bearing and not a transliteration choice: each walk starts from the
// point the previous one refined, so blue measures from red's corrected centre
// and the second u walk measures from the v walk's. Running the two channels
// independently from the raw seed instead costs the low-module-scale rows,
// where a seed pixel is a large fraction of a module and one channel's walk
// alone does not converge.
func crossCheckPatternAPBasis(ch [3]*core.Bitmap, b apBasis, apType int, moduleSizeMax float64,
	centre *core.PointF, moduleSize *float64, dir *int, back, fwd int,
) bool {
	var msU, msV [3]float64

	pR := *centre
	if !crossCheckAPUAlong(ch[0], b, apType, 0, moduleSizeMax, &pR, &msU[0], back, fwd) {
		return false
	}
	pB := pR
	if !crossCheckAPUAlong(ch[2], b, apType, 2, moduleSizeMax, &pB, &msU[2], back, fwd) {
		return false
	}
	p := midpoint(pR, pB)
	*moduleSize = (msU[0] + msU[2]) / 2.0

	greenCore := int(palette.Default[apCoreColorIndex(apType)*3+1])
	if !crossCheckColorAPBasis(ch[1], greenCore, *moduleSize, p, b) {
		return false
	}

	pR = p
	if !crossCheckAPAlong(ch[0], b.v, moduleSizeMax, b.v.pxPerSample, &pR, &msV[0], unboundedWalk, unboundedWalk) {
		return false
	}
	if !crossCheckAPUAlong(ch[0], b, apType, 0, moduleSizeMax, &pR, &msU[0], back, fwd) {
		return false
	}
	pB = p
	if !crossCheckAPAlong(ch[2], b.v, moduleSizeMax, b.v.pxPerSample, &pB, &msV[2], unboundedWalk, unboundedWalk) {
		return false
	}
	if !crossCheckAPUAlong(ch[2], b, apType, 2, moduleSizeMax, &pB, &msU[2], back, fwd) {
		return false
	}

	*moduleSize = (msU[0] + msU[2] + msV[0] + msV[2]) / 4.0
	p = midpoint(pR, pB)
	if !crossCheckColorAPBasis(ch[1], greenCore, *moduleSize, p, b) {
		return false
	}

	dirR, dirB := 0, 0
	var msD float64
	probe := p
	if !crossCheckAPDiagonalBasis(ch[0], b, apType, moduleSizeMax, &probe, &msD, &dirR) {
		return false
	}
	probe = p
	if !crossCheckAPDiagonalBasis(ch[2], b, apType, moduleSizeMax, &probe, &msD, &dirB) {
		return false
	}
	if !crossCheckColorAPBasis(ch[1], greenCore, *moduleSize, p, b) {
		return false
	}

	*centre = p
	if dirR+dirB > 0 {
		*dir = 1
	} else {
		*dir = -1
	}
	return true
}

// findAlignmentPatternBasis searches for an alignment pattern of the given type
// near (x, y), sweeping lines of the local basis instead of image rows.
//
// The sweep shape is the axis-aligned one with both axes replaced: lines run
// along u and are offset along v, alternating outward from the predicted
// position so a correct prediction is confirmed on the first line, and the
// window doubles on failure over the same range.
func findAlignmentPatternBasis(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int, b apBasis) FinderPattern {
	coreColorR := palette.Default[apCoreColorIndex(apType)*3]
	ux, uy := b.u.unit()
	vx, vy := b.v.unit()

	sample := func(px, py float64) (byte, bool) {
		ix, iy := int(px), int(py)
		if ix < 0 || ix >= ch[0].Width || iy < 0 || iy >= ch[0].Height {
			return 0, false
		}
		return ch[0].Pixel(ix, iy), true
	}

	radius := int(4 * moduleSize)
	radiusMax := 4 * radius
	for ; radius < radiusMax; radius <<= 1 {
		if float64(radius) < 1.5*moduleSize {
			continue
		}
		aps := make([]FinderPattern, maxFinderPatterns)
		counter := 0
		for kk := range 2 * radius {
			k := (kk + 1) / 2
			if kk&0x01 != 0 {
				k = -k
			}
			lx := x + float64(k)*vx
			ly := y + float64(k)*vy

			// Two cursors run outward along u from the predicted point, taking
			// turns, each stopping at the start of the next run of core colour.
			// The line ends at its first accepted candidate.
			//
			// Both properties are load-bearing rather than cosmetic. Alternating
			// keeps the candidates ordered by distance from the prediction, and
			// stopping at the first acceptance keeps one line from contributing
			// several. A sweep that instead cross-checks every core run on every
			// line accepts hundreds of candidates on a low-resolution symbol, and
			// then the first pair close enough to merge is a spurious one rather
			// than the predicted pattern.
			cur := [2]struct {
				i    int
				sign float64
				run  bool
				done bool
			}{{sign: -1}, {sign: +1}}
			accepted := false
			for side := 0; !accepted && !(cur[0].done && cur[1].done); side ^= 1 {
				c := &cur[side]
				if c.done {
					continue
				}
				for !c.done {
					if c.i > radius {
						c.done = true
						break
					}
					px := lx + c.sign*float64(c.i)*ux
					py := ly + c.sign*float64(c.i)*uy
					c.i++
					v, ok := sample(px, py)
					if !ok {
						c.done = true
						break
					}
					if v != coreColorR {
						c.run = false
						continue
					}
					if c.run {
						continue
					}
					c.run = true

					centre := core.Pt(px, py)
					var apModuleSize float64
					apDir := 0
					// The seed's own distance to the window edge along u, which is
					// what the axis-aligned walk gets from startx and endx.
					off := int(c.sign) * (c.i - 1)
					if !crossCheckPatternAPBasis(ch, b, apType, moduleSize*2, &centre, &apModuleSize, &apDir, radius+off, radius-off) {
						continue
					}
					ap := FinderPattern{Typ: apType, FoundCount: 1, ModuleSize: apModuleSize, Center: centre, direction: apDir}
					if index := saveAlignmentPattern(&ap, aps, &counter); index >= 0 {
						return aps[index]
					}
					if counter >= maxFinderPatterns {
						return FinderPattern{Typ: -1}
					}
					accepted = true
					break
				}
			}
		}
	}
	return FinderPattern{Typ: -1}
}
