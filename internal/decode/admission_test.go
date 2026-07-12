package decode

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// admissionObserve renders a symbol at the given colour count, builds its
// pixel-exact one-module-per-pixel bitmap with the module grid rolled by
// shift modules on both axes (0 = true grid, nonzero = a misgrid analog),
// and observes it. The payload is long enough that the symbol carries
// interior alignment patterns, so the fixed-pattern walk exercises both its
// finder and its alignment halves.
func admissionObserve(t *testing.T, colors, shift int) *PrimaryObservation {
	t.Helper()
	payload := bytes.Repeat([]byte("admission signal test payload "), 8)
	r, err := encode.Render(encode.Config{Colors: colors, ModuleSize: 1, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("render %dc: %v", colors, err)
	}
	w, h := r.SideSize.X, r.SideSize.Y
	bm := core.NewBitmap(w, h, 4)
	for y := range h {
		for x := range w {
			idx := r.Matrix[((y+shift)%h)*w+(x+shift)%w]
			copy(bm.Pix[(y*w+x)*4:], r.Palette[int(idx)*3:int(idx)*3+3])
			bm.Pix[(y*w+x)*4+3] = 255
		}
	}
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("%dc shift %d: ObservePrimary => %d", colors, shift, ret)
	}
	return obs
}

func TestFixedPatternAgreementSeparatesGrids(t *testing.T) {
	for _, colors := range []int{8, 32} {
		trueObs := admissionObserve(t, colors, 0)
		agree, checked := trueObs.FixedPatternAgreement()
		if checked < 68+7 {
			// 4 finders x 17 modules plus at least one interior alignment
			// pattern (core + 6 periphery).
			t.Errorf("%dc: only %d fixed modules checked", colors, checked)
		}
		if agree != checked {
			t.Errorf("%dc true grid: %d/%d fixed modules agree", colors, agree, checked)
		}

		badObs := admissionObserve(t, colors, 3)
		bAgree, bChecked := badObs.FixedPatternAgreement()
		if bChecked < 68 {
			t.Errorf("%dc shifted: only %d fixed modules checked", colors, bChecked)
		}
		if bAgree*2 >= bChecked {
			t.Errorf("%dc shifted grid: %d/%d fixed modules agree, want under half", colors, bAgree, bChecked)
		}
	}
}

func TestPaletteCoherenceOnTrueGrid(t *testing.T) {
	for _, colors := range []int{8, 32} {
		obs := admissionObserve(t, colors, 0)
		disagreement, separation := obs.PaletteCoherence()
		if disagreement != 0 {
			t.Errorf("%dc: pixel-exact copies disagree by %.2f", colors, disagreement)
		}
		if separation <= 0 {
			t.Errorf("%dc: palette separation %.2f, want positive", colors, separation)
		}
		if disagreement >= separation {
			t.Errorf("%dc: disagreement %.2f not below separation %.2f", colors, disagreement, separation)
		}
	}
}
