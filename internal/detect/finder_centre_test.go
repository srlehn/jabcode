package detect

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// centreSection reports an exact 2m/m/m/m/2m finder cross section as a function
// of signed distance from the middle run's centre pixel. m is odd so the middle
// run is symmetric about that pixel and every walk direction agrees on where its
// centre lies; the refined value is then the centre pixel's low edge plus half a
// run, which is that pixel's midpoint.
func centreSection(m, d int) byte {
	const (
		dark  = 0
		light = 255
	)
	h := (m - 1) / 2
	switch a := abs(d); {
	case a <= h:
		return light
	case a <= h+m:
		return dark
	case a <= h+3*m:
		return light
	}
	return dark
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// centreCrossFixture carries the section along both axes. The axis walks read
// only the row through cy and the column through cx, where the two sections
// cannot disagree, so each sees the exact runs.
func centreCrossFixture(m, cx, cy, size int) *core.Bitmap {
	bm := &core.Bitmap{Width: size, Height: size, Channels: 1, Pix: make([]byte, size*size)}
	for y := range size {
		for x := range size {
			bm.Pix[y*size+x] = min(centreSection(m, x-cx), centreSection(m, y-cy))
		}
	}
	return bm
}

// centreDiagonalFixture carries the section as bands perpendicular to one of the
// two finder diagonals, so a walk along that diagonal advances exactly one run
// step per sample and the true centre is the pixel at (cx, cy) for both classes.
func centreDiagonalFixture(m, cx, cy, size int, anti bool) *core.Bitmap {
	bm := &core.Bitmap{Width: size, Height: size, Channels: 1, Pix: make([]byte, size*size)}
	for y := range size {
		for x := range size {
			d := (x - cx) + (y - cy)
			if anti {
				d = (y - cy) - (x - cx)
			}
			bm.Pix[y*size+x] = centreSection(m, d>>1)
		}
	}
	return bm
}

// The diagonal walk confirms along startx-i*offsetX but derived centerx from
// startx+i, so wherever offsetX is +1 - fp2, fp3, and any type the caller pins
// to that diagonal - the refinement was reflected about the seed instead of
// applied to it. Each seed sits on the diagonal its walk will take, so every
// case here has the same reachable true centre.
func TestDiagonalCentreRefinementFollowsWalkDirection(t *testing.T) {
	const (
		m    = 7
		size = 121
		cx   = 60
		cy   = 60
	)
	want := cx + 0.5

	for _, tc := range []struct {
		name string
		typ  int
		dir  int
		seed int
	}{
		{"fp0 seeded ahead", fp0, 0, +2},
		{"fp0 seeded behind", fp0, 0, -2},
		{"fp1 seeded ahead", fp1, 0, +2},
		{"fp2 seeded ahead", fp2, 0, +2},
		{"fp2 seeded behind", fp2, 0, -2},
		{"fp3 seeded ahead", fp3, 0, +2},
		{"fp0 pinned to the fp2 diagonal", fp0, -1, +2},
		{"fp2 pinned to the fp0 diagonal", fp2, +1, +2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A walk can only correct along its own line, so the seed is
			// displaced along that line: y follows x on the fp0/fp1 diagonal
			// and opposes it on the fp2/fp3 one.
			down := tc.typ == fp0 || tc.typ == fp1
			if tc.dir != 0 {
				down = tc.dir > 0
			}
			bm := centreDiagonalFixture(m, cx, cy, size, !down)
			centrex, centrey := float64(cx+tc.seed), float64(cy-tc.seed)
			if down {
				centrey = float64(cy + tc.seed)
			}

			dir := tc.dir
			var module float64
			if got := crossCheckPatternDiagonal(bm, tc.typ, 2*m, &centrex, &centrey, &module, &dir, false, 3); got == 0 {
				t.Fatalf("diagonal did not confirm, module=%v", module)
			}
			if module != m {
				t.Errorf("module size = %v, want %d", module, m)
			}
			if centrex != want || centrey != want {
				t.Errorf("refined centre = (%v, %v), want (%v, %v)", centrex, centrey, want, want)
			}
		})
	}
}

// Both axis walks end with the same expression, so the half pixel they share is
// the convention that a coordinate names the low edge of a pixel cell. The
// vertical walk must not add a second one on top: it charges the centre pixel
// into the middle run exactly as the horizontal walk does.
func TestAxisCentreRefinementAgreesBetweenWalks(t *testing.T) {
	const (
		size = 121
		cx   = 60
		cy   = 60
	)
	want := cx + 0.5
	for _, m := range []int{5, 7, 9} {
		bm := centreCrossFixture(m, cx, cy, size)
		for _, seed := range []int{-2, 0, +2} {
			centrex, centrey := float64(cx+seed), float64(cy+seed)
			var moduleH, moduleV float64
			if !crossCheckPatternHorizontal(bm, float64(2*m), &centrex, float64(cy), &moduleH, 3) {
				t.Fatalf("m=%d seed=%+d: horizontal did not confirm", m, seed)
			}
			if !crossCheckPatternVertical(bm, 2*m, float64(cx), &centrey, &moduleV, 3) {
				t.Fatalf("m=%d seed=%+d: vertical did not confirm", m, seed)
			}
			if moduleH != float64(m) || moduleV != float64(m) {
				t.Errorf("m=%d seed=%+d: module sizes %v and %v", m, seed, moduleH, moduleV)
			}
			if centrex != want || centrey != want {
				t.Errorf("m=%d seed=%+d: refined centres %v and %v, want %v",
					m, seed, centrex, centrey, want)
			}
		}
	}
}
