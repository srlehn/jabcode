package decode

import "testing"

// stripeBitmap renders a 4-channel square-wave test pattern: a vertical-stripe
// component of period periodX (0 = none) XOR'd with a horizontal-stripe component
// of period periodY (0 = none). With one period zero the result is pure stripes
// along that axis, the clean case for checking the per-axis pitch estimate.
func stripeBitmap(w, h, periodX, periodY int) *Bitmap {
	bm := NewBitmap(w, h, 4)
	for y := range h {
		for x := range w {
			on := true
			if periodX > 0 && x%periodX >= periodX/2 {
				on = !on
			}
			if periodY > 0 && y%periodY >= periodY/2 {
				on = !on
			}
			var v byte
			if on {
				v = 255
			}
			o := (y*w + x) * 4
			bm.Pix[o], bm.Pix[o+1], bm.Pix[o+2], bm.Pix[o+3] = v, v, v, 255
		}
	}
	return bm
}

func TestEstimatePitchVerticalStripes(t *testing.T) {
	for _, period := range []int{4, 8, 12, 20} {
		bm := stripeBitmap(640, 480, period, 0)
		px, py := EstimatePitch(bm)
		if px != period {
			t.Errorf("vertical stripes period %d: px = %d, want %d", period, px, period)
		}
		if py != 0 {
			t.Errorf("vertical stripes period %d: py = %d, want 0 (no vertical periodicity)", period, py)
		}
	}
}

func TestEstimatePitchHorizontalStripes(t *testing.T) {
	for _, period := range []int{6, 10, 16} {
		bm := stripeBitmap(640, 480, 0, period)
		px, py := EstimatePitch(bm)
		if py != period {
			t.Errorf("horizontal stripes period %d: py = %d, want %d", period, py, period)
		}
		if px != 0 {
			t.Errorf("horizontal stripes period %d: px = %d, want 0 (no horizontal periodicity)", period, px)
		}
	}
}

func TestEstimatePitchFlatIsZero(t *testing.T) {
	bm := stripeBitmap(320, 240, 0, 0) // all-on, no periodicity
	px, py := EstimatePitch(bm)
	if px != 0 || py != 0 {
		t.Errorf("flat image: (px,py) = (%d,%d), want (0,0)", px, py)
	}
}

func TestEstimatePitchTinyImage(t *testing.T) {
	bm := stripeBitmap(3, 3, 2, 0)
	if px, py := EstimatePitch(bm); px != 0 || py != 0 {
		t.Errorf("tiny image: (px,py) = (%d,%d), want (0,0)", px, py)
	}
}
