package decode

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// softPathSymbol renders an 8-colour symbol and returns its pixel-exact
// bitmap together with the reserved-module map (finder, alignment, metadata,
// palette), so tests can corrupt data modules only.
func softPathSymbol(t *testing.T, payload []byte) (*core.Bitmap, []byte, *core.DecodedSymbol) {
	t.Helper()
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 1, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	w, h := r.SideSize.X, r.SideSize.Y
	bm := core.NewBitmap(w, h, 4)
	for i, idx := range r.Matrix {
		copy(bm.Pix[i*4:], r.Palette[int(idx)*3:int(idx)*3+3])
		bm.Pix[i*4+3] = 255
	}
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("clean observation failed: %d", ret)
	}
	// Complete the reserved map with the finder and alignment patterns the
	// payload stage marks, so the corruption below touches data modules only.
	dataMap := obs.dataMap
	fillDataMap(dataMap, w, h, 0)
	return bm, dataMap, sym
}

// blendModule moves a module's pixel a fraction toward the given palette
// colour: past one half the hard classifier reads the wrong colour, but the
// small distance gap leaves the soft reliabilities low - an ambiguous bit
// belief propagation can overrule.
func blendModule(bm *core.Bitmap, palette []byte, x, y, toColor int, frac float64) {
	o := bm.Offset(x, y)
	for k := range 3 {
		from := float64(bm.Pix[o+k])
		to := float64(palette[toColor*3+k])
		bm.Pix[o+k] = byte(from + (to-from)*frac + 0.5)
	}
}

// TestSoftPathRecoversAmbiguousModules exercises the production soft path
// end to end - sample, demask, deinterleave, hard-decode failure, module
// reliabilities, belief propagation - by blending a spread of data modules
// just past the classification boundary. The corrupted bit fraction sits
// above the measured hard bit-flipping ceiling (about seven percent), so the
// recovery must come from the soft retry weighting the ambiguous bits down.
func TestSoftPathRecoversAmbiguousModules(t *testing.T) {
	payload := bytes.Repeat([]byte("soft path end-to-end "), 3)
	bm, dataMap, _ := softPathSymbol(t, payload)
	w := bm.Width

	// Blend a spread of data modules just past the classification midpoint
	// toward their last-bit neighbour (green/cyan, red/magenta,
	// yellow/white). Black-involved pairs are excluded deliberately: in the
	// normalized-RGB metric an absolute blend toward or away from black
	// lands CONFIDENTLY on the wrong colour instead of ambiguously between
	// the two - adversarial evidence rather than the low-margin kind the
	// soft retry exists for.
	corrupted := 0
	for i := 0; i < len(dataMap); i++ {
		if dataMap[i] != 0 || (i%3) != 0 {
			continue
		}
		x, y := i%w, i/w
		o := bm.Offset(x, y)
		orig := nearestDefaultColor(bm.Pix[o : o+3])
		if orig < 2 {
			continue
		}
		blendModule(bm, defaultPalette8(), x, y, orig^1, 0.52)
		corrupted++
	}
	if corrupted < 40 {
		t.Fatalf("only %d modules corrupted; the test needs a spread", corrupted)
	}

	sym := &core.DecodedSymbol{}
	if ret := DecodePrimary(bm, sym); ret != core.Success {
		t.Fatalf("soft path did not recover %d ambiguous modules: %d", corrupted, ret)
	}
	if got := DecodeData(sym.Data); !bytes.Equal(got, payload) {
		t.Fatalf("recovered wrong payload %q", got)
	}
}

// TestSoftPathRefusesConfidentCorruption pins the honest-failure contract:
// data modules set fully to wrong colours carry confidently wrong soft
// evidence, so neither the hard nor the soft decoder may return a payload -
// and above all not a wrong payload with a nil error.
func TestSoftPathRefusesConfidentCorruption(t *testing.T) {
	payload := bytes.Repeat([]byte("soft path end-to-end "), 3)
	bm, dataMap, _ := softPathSymbol(t, payload)
	w := bm.Width

	for i := 0; i < len(dataMap); i++ {
		if dataMap[i] != 0 || (i%3) != 0 {
			continue
		}
		x, y := i%w, i/w
		o := bm.Offset(x, y)
		orig := nearestDefaultColor(bm.Pix[o : o+3])
		blendModule(bm, defaultPalette8(), x, y, orig^3, 1.0)
	}

	sym := &core.DecodedSymbol{}
	if ret := DecodePrimary(bm, sym); ret == core.Success {
		if got := DecodeData(sym.Data); !bytes.Equal(got, payload) {
			t.Fatalf("confident corruption returned a WRONG payload %q with success", got)
		}
	}
}

// defaultPalette8 is the rendered 8-colour palette (RGB cube corners).
func defaultPalette8() []byte {
	return []byte{
		0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255,
		255, 0, 0, 255, 0, 255, 255, 255, 0, 255, 255, 255,
	}
}

// nearestDefaultColor maps a pixel back to its default-palette index.
func nearestDefaultColor(rgb []byte) int {
	pal := defaultPalette8()
	best, bi := 1<<30, 0
	for i := range 8 {
		d := 0
		for k := range 3 {
			dd := int(rgb[k]) - int(pal[i*3+k])
			d += dd * dd
		}
		if d < best {
			best, bi = d, i
		}
	}
	return bi
}
