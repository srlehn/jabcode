package detect

import (
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// renderRotatedSymbol draws a real encoded symbol under an exact projective map
// and returns the binarized green channel plus the forward transform. Every
// output pixel takes the colour of the module it falls in, so there is no
// resampling blur and any detection failure belongs to the scanner.
func renderRotatedSymbol(t *testing.T, r encode.Rendered, modulePx, theta float64) (*core.Bitmap, core.Perspective) {
	t.Helper()
	sx, sy := float64(r.SideSize.X), float64(r.SideSize.Y)
	w, h := sx*modulePx, sy*modulePx

	src := [4]core.PointF{{X: 0, Y: 0}, {X: sx, Y: 0}, {X: sx, Y: sy}, {X: 0, Y: sy}}
	corners := [4]core.PointF{{X: 0, Y: 0}, {X: w, Y: 0}, {X: w, Y: h}, {X: 0, Y: h}}
	cx, cy := w/2, h/2
	ct, st := math.Cos(theta), math.Sin(theta)
	var dst [4]core.PointF
	minX, minY := math.Inf(1), math.Inf(1)
	for i, p := range corners {
		dx, dy := p.X-cx, p.Y-cy
		dst[i] = core.PointF{X: cx + dx*ct - dy*st, Y: cy + dx*st + dy*ct}
		minX, minY = math.Min(minX, dst[i].X), math.Min(minY, dst[i].Y)
	}
	const margin = 24.0
	maxX, maxY := 0.0, 0.0
	for i := range dst {
		dst[i].X += margin - minX
		dst[i].Y += margin - minY
		maxX, maxY = math.Max(maxX, dst[i].X), math.Max(maxY, dst[i].Y)
	}
	fwd := core.QuadToQuad(src, dst)
	inv := core.QuadToQuad(dst, src)

	bw, bh := int(maxX+margin), int(maxY+margin)
	green := core.NewBitmap(bw, bh, 1)
	for y := range bh {
		for x := range bw {
			v := byte(255)
			m := inv.Warp(core.Pt(float64(x)+0.5, float64(y)+0.5))
			if m.X >= 0 && m.X < sx && m.Y >= 0 && m.Y < sy {
				idx := int(r.Matrix[int(m.Y)*r.SideSize.X+int(m.X)])
				if r.Palette[3*idx+1] <= 127 {
					v = 0
				}
			}
			green.Pix[y*bw+x] = v
		}
	}
	return green, fwd
}

func directionalTestSymbol(t *testing.T) encode.Rendered {
	t.Helper()
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 1, ECCLevel: 5, SymbolNumber: 1},
		[]byte("directional finder scan"))
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return r
}

// finderCentres returns the four finder cores in image coordinates.
func finderCentres(fwd core.Perspective, side [2]float64) [4]core.PointF {
	uv := [4]core.PointF{
		{X: 3.5, Y: 3.5}, {X: side[0] - 3.5, Y: 3.5},
		{X: side[0] - 3.5, Y: side[1] - 3.5}, {X: 3.5, Y: side[1] - 3.5},
	}
	var out [4]core.PointF
	for i, p := range uv {
		out[i] = fwd.Warp(p)
	}
	return out
}

// TestCrossCheckAlongAcceptsRotatedFinders is the load-bearing test for
// directional scanning: at every orientation there must be a scan direction
// that confirms all four finders, including the angles where the axis-aligned
// walk sees nothing. Without this the ladder cannot be removed.
func TestCrossCheckAlongAcceptsRotatedFinders(t *testing.T) {
	r := directionalTestSymbol(t)
	side := [2]float64{float64(r.SideSize.X), float64(r.SideSize.Y)}
	const modulePx = 12.0

	for _, deg := range []float64{0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90, 135, 200, 285, 330} {
		th := deg * math.Pi / 180
		green, fwd := renderRotatedSymbol(t, r, modulePx, th)
		centres := finderCentres(fwd, side)

		best, bestDir := 0, -1.0
		for _, probe := range scanDirections {
			// A scan folds modulo 180 and the symbol's two axes are 90 apart, so
			// probing [0,90) covers every orientation.
			dir := newScanDirection(probe)
			confirmed := 0
			for _, c := range centres {
				centre := c
				var ms float64
				if crossCheckPatternAlong(green, dir, modulePx*2, &centre, &ms, 3) {
					confirmed++
				}
			}
			if confirmed > best {
				best, bestDir = confirmed, probe
			}
		}
		if best < 4 {
			t.Errorf("theta=%3.0f: best direction %v confirmed only %d of 4 finders", deg, bestDir, best)
			continue
		}
		t.Logf("theta=%3.0f: all 4 finders confirmed at scan direction %.0f", deg, bestDir)
	}
}

// TestCrossCheckAlongReportsPhysicalModuleSize pins the sample-to-pixel
// conversion. A directional walk measures runs in samples, each covering
// 1/max(|cos|,|sin|) pixels, so without the conversion the module size would
// read up to 29 percent short at 45 degrees and every downstream consumer of
// ModuleSize would be wrong.
func TestCrossCheckAlongReportsPhysicalModuleSize(t *testing.T) {
	r := directionalTestSymbol(t)
	side := [2]float64{float64(r.SideSize.X), float64(r.SideSize.Y)}
	const modulePx = 12.0

	for _, deg := range []float64{0, 15, 30, 45, 60, 75} {
		th := deg * math.Pi / 180
		green, fwd := renderRotatedSymbol(t, r, modulePx, th)
		centres := finderCentres(fwd, side)
		dir := newScanDirection(deg)

		var sum float64
		n := 0
		for _, c := range centres {
			centre := c
			var ms float64
			if crossCheckPatternAlong(green, dir, modulePx*2, &centre, &ms, 3) {
				sum += ms
				n++
			}
		}
		if n == 0 {
			t.Fatalf("theta=%3.0f: no finder confirmed, cannot check module size", deg)
		}
		mean := sum / float64(n)
		// The scan runs along a symbol axis here, so the measured module size is
		// the true one once samples are converted to pixels.
		if math.Abs(mean-modulePx) > 0.25*modulePx {
			t.Errorf("theta=%3.0f: module size %.2f is not within 25%% of the true %.1f", deg, mean, modulePx)
		}
		t.Logf("theta=%3.0f: module size %.2f (true %.1f) over %d finders", deg, mean, modulePx, n)
	}
}

// TestSeekPatternAlongFindsRotatedFinders is the traversal half: the scan must
// discover the finders by sweeping lines over the frame, not by being handed
// their positions. This is what replaces the row walk.
//
// It proves the directional traversal works, not that it is what rotation
// needed. Raw run-length hits are rotation-robust by design, so several
// directions find the signature at any orientation - which is why the winning
// direction here is often not the aligned one. The stage that actually
// discriminates orientation is the cross-check, covered above.
func TestSeekPatternAlongFindsRotatedFinders(t *testing.T) {
	r := directionalTestSymbol(t)
	side := [2]float64{float64(r.SideSize.X), float64(r.SideSize.Y)}
	const modulePx = 12.0

	for _, deg := range []float64{0, 30, 45, 60, 135, 200, 330} {
		th := deg * math.Pi / 180
		green, fwd := renderRotatedSymbol(t, r, modulePx, th)
		centres := finderCentres(fwd, side)

		bestHit, bestDir := 0, -1.0
		for _, probe := range scanDirections {
			dir := newScanDirection(probe)
			hits := sweepDirectionForTest(green, dir, 1)
			near := 0
			for _, c := range centres {
				for _, h := range hits {
					if math.Hypot(h.X-c.X, h.Y-c.Y) <= modulePx {
						near++
						break
					}
				}
			}
			if near > bestHit {
				bestHit, bestDir = near, probe
			}
		}
		if bestHit < 4 {
			t.Errorf("theta=%3.0f: best direction %v found only %d of 4 finders by scanning", deg, bestDir, bestHit)
			continue
		}
		t.Logf("theta=%3.0f: all 4 finders found by scanning at direction %.0f", deg, bestDir)
	}
}

// sweepDirectionForTest covers the frame with parallel lines at dir, spaced
// step apart perpendicular to the scan, and collects every run-length
// signature. It is the shape the family scan will take.
func sweepDirectionForTest(img *core.Bitmap, dir scanDirection, step int) []core.PointF {
	var out []core.PointF
	w, h := float64(img.Width), float64(img.Height)
	// Lines are indexed by perpendicular offset from the origin. The unit
	// perpendicular is (-dy,dx) scaled by pxPerSample, since |(dx,dy)| is
	// exactly pxPerSample for a major-axis step.
	nx, ny := -dir.dy/dir.pxPerSample, dir.dx/dir.pxPerSample
	span := w + h
	limit := int(2*span) + 2
	for q := -span; q < span; q += float64(step) {
		x0, y0 := q*nx, q*ny
		start := -limit / 2
		// A scan line crosses many data-module run patterns before reaching a
		// finder, so the per-line hit budget has to be generous; the production
		// row walk scans the whole row for the same reason.
		for hits := 0; hits < 4096; hits++ {
			centre, ms, _, next, ok := seekPatternAlong(img, dir, x0, y0, start, limit)
			if !ok {
				break
			}
			if ms > 0 {
				out = append(out, centre)
			}
			start = next
		}
	}
	return out
}

// TestScanDirectionSampleGeometry pins the two geometric facts the whole design
// rests on: one sample covers 1/max(|cos|,|sin|) pixels, and the perpendicular
// direction is a quarter turn with the same property.
func TestScanDirectionSampleGeometry(t *testing.T) {
	for _, deg := range []float64{0, 15, 30, 45, 60, 75, 90} {
		d := newScanDirection(deg)
		a := deg * math.Pi / 180
		want := 1 / math.Max(math.Abs(math.Cos(a)), math.Abs(math.Sin(a)))
		if math.Abs(d.pxPerSample-want) > 1e-12 {
			t.Errorf("deg=%v pxPerSample=%v want %v", deg, d.pxPerSample, want)
		}
		if maj := math.Max(math.Abs(d.dx), math.Abs(d.dy)); math.Abs(maj-1) > 1e-12 {
			t.Errorf("deg=%v major axis step %v, want 1", deg, maj)
		}
		p := d.perpendicular()
		if dot := d.dx*p.dx + d.dy*p.dy; math.Abs(dot) > 1e-9 {
			t.Errorf("deg=%v perpendicular dot %v, want 0", deg, dot)
		}
	}
}
