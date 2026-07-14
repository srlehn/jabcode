package decode

import (
	"bytes"
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
)

// TestModuleCostsAndBitLLRs pins the signed-evidence contract on a clean
// 8-colour observation: every data module's own colour carries the minimum
// finite cost (including black, via the luminance term), and the bit LLR
// signs follow the tested convention - positive favors bit zero.
func TestModuleCostsAndBitLLRs(t *testing.T) {
	bm, dataMap, _ := softPathSymbol(t, []byte("signed evidence contract"))
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observation failed: %d", ret)
	}
	w := bm.Width
	seen := 0
	for i := 0; i < len(dataMap) && seen < 64; i++ {
		if dataMap[i] != 0 {
			continue
		}
		x, y := i%w, i/w
		o := bm.Offset(x, y)
		want := nearestDefaultColor(bm.Pix[o : o+3])
		costs := obs.ModuleCosts(x, y, nil)
		if len(costs) != 8 {
			t.Fatalf("(%d,%d): %d costs, want 8", x, y, len(costs))
		}
		bestC, best := math.Inf(1), -1
		for c, v := range costs {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("(%d,%d): cost[%d] = %v, want finite", x, y, c, v)
			}
			if v < bestC {
				bestC, best = v, c
			}
		}
		if best != want {
			t.Fatalf("(%d,%d): minimum cost at colour %d, want %d", x, y, best, want)
		}
		llrs := BitLLRs(costs, nil)
		if len(llrs) != 3 {
			t.Fatalf("(%d,%d): %d LLRs, want 3", x, y, len(llrs))
		}
		for p, l := range llrs {
			bit := (want >> uint(2-p)) & 1
			if bit == 1 && l >= 0 {
				t.Errorf("(%d,%d): bit %d is set but LLR %.3f is not negative", x, y, p, l)
			}
			if bit == 0 && l <= 0 {
				t.Errorf("(%d,%d): bit %d is clear but LLR %.3f is not positive", x, y, p, l)
			}
		}
		seen++
	}
	if seen == 0 {
		t.Fatal("no data modules checked")
	}
}

// TestModuleCostsBlackAmbiguity pins the measured black constraint: a module
// blended from black just past the midpoint toward blue must cost black and
// blue comparably (an ambiguous, low-margin decision), not land confidently
// on blue the way the pure chroma metric does.
func TestModuleCostsBlackAmbiguity(t *testing.T) {
	bm, dataMap, _ := softPathSymbol(t, []byte("signed evidence contract"))
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observation failed: %d", ret)
	}
	w := bm.Width
	var x, y int
	for i := 0; i < len(dataMap); i++ {
		if dataMap[i] == 0 {
			x, y = i%w, i/w
			break
		}
	}
	o := bm.Offset(x, y)
	bm.Pix[o], bm.Pix[o+1], bm.Pix[o+2] = 0, 0, 132 // black blended 52% toward blue

	costs := obs.ModuleCosts(x, y, nil)
	first, second := 0, 1
	if costs[second] < costs[first] {
		first, second = second, first
	}
	for c := 2; c < len(costs); c++ {
		switch {
		case costs[c] < costs[first]:
			first, second = c, first
		case costs[c] < costs[second]:
			second = c
		}
	}
	if (first != 0 || second != 1) && (first != 1 || second != 0) {
		t.Fatalf("blended module ranks colours %d,%d first, want black and blue", first, second)
	}
	if gap, span := costs[second]-costs[first], costs[7]-costs[first]; gap > span/4 {
		t.Errorf("black/blue gap %.4f not ambiguous against span %.4f", gap, span)
	}
}

// TestSnapshotBitEvidenceMatchesTruth pins the gross-coordinate contract: a
// clean snapshot's signed evidence must agree, bit for bit through demask,
// truncation and deinterleave, with the demasked module bits the encoder
// actually placed - negative wherever the true bit is one, positive where
// zero.
func TestSnapshotBitEvidenceMatchesTruth(t *testing.T) {
	payload := bytes.Repeat([]byte("gross evidence "), 3)
	bm, _, _ := softPathSymbol(t, payload)
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 1, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observation failed: %d", ret)
	}
	snap := obs.Snapshot()
	ev := snap.BitEvidence()
	if ev == nil {
		t.Fatal("no evidence derived")
	}

	w, h := snap.Side.X, snap.Side.Y
	var want []byte
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			if snap.DataMap[y*w+x] != 0 {
				continue
			}
			idx := int(r.Matrix[y*w+x]) ^ (spec.MaskValue(snap.Meta.MaskType, x, y) % 8)
			for p := 2; p >= 0; p-- {
				want = append(want, byte((idx>>uint(p))&1))
			}
		}
	}
	if len(ev) > len(want) {
		t.Fatalf("evidence longer than the bit stream: %d > %d", len(ev), len(want))
	}
	want = want[:len(ev)]
	ecc.Deinterleave(want)
	for i, l := range ev {
		if want[i] == 1 && l >= 0 {
			t.Fatalf("bit %d is one but evidence %.4f is not negative", i, l)
		}
		if want[i] == 0 && l <= 0 {
			t.Fatalf("bit %d is zero but evidence %.4f is not positive", i, l)
		}
	}
	retained := append([]float64(nil), ev...)
	corrected, ret := snap.CorrectEvidence(ev)
	if ret != core.Success || corrected == nil {
		t.Fatalf("correct accumulated evidence: %d", ret)
	}
	wantPayload := append([]byte("]j1"), payload...)
	if got := DecodeData(corrected.Data); !bytes.Equal(got, wantPayload) {
		t.Fatalf("corrected evidence payload = %q, want %q", got, wantPayload)
	}
	for i := range ev {
		if math.Float64bits(ev[i]) != math.Float64bits(retained[i]) {
			t.Fatalf("correction mutated retained evidence at %d", i)
		}
	}
}

// TestSnapshotBitEvidenceRequiresTrustedLayout pins the mode-specific trust
// rule. Explicit metadata may authorize bit-coordinate evidence only when
// both metadata parts satisfied their syndromes. Default mode has no explicit
// metadata parts and is authorized by its structurally defined layout.
func TestSnapshotBitEvidenceRequiresTrustedLayout(t *testing.T) {
	bm, _, _ := softPathSymbol(t, []byte("trusted evidence layout"))
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observation failed: %d", ret)
	}
	snap := obs.Snapshot()
	if snap.Meta.DefaultMode {
		t.Fatal("test fixture unexpectedly used default metadata")
	}
	if len(snap.BitEvidence()) == 0 {
		t.Fatal("trusted explicit metadata produced no evidence")
	}

	snap.PartISyndromeOK = false
	if got := snap.BitEvidence(); got != nil {
		t.Fatalf("failed part I syndrome produced %d evidence values", len(got))
	}
	snap.PartISyndromeOK = true
	snap.PartIISyndromeOK = false
	if got := snap.BitEvidence(); got != nil {
		t.Fatalf("failed part II syndrome produced %d evidence values", len(got))
	}

	// Merely relabeling explicit metadata as default must not authorize it.
	snap.Meta.DefaultMode = true
	if got := snap.BitEvidence(); got != nil {
		t.Fatalf("invalid default configuration produced %d evidence values", len(got))
	}

	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 1, ECCLevel: spec.DefaultECCLevel, SymbolNumber: 1}, []byte("default evidence layout"))
	if err != nil {
		t.Fatalf("render default symbol: %v", err)
	}
	defaultBM := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, idx := range r.Matrix {
		copy(defaultBM.Pix[i*4:], r.Palette[int(idx)*3:int(idx)*3+3])
		defaultBM.Pix[i*4+3] = 255
	}
	defaultSym := &core.DecodedSymbol{}
	defaultObs, ret := ObservePrimary(defaultBM, defaultSym)
	if ret != core.Success || defaultObs == nil || !defaultSym.Meta.DefaultMode {
		t.Fatalf("default observation failed: ret=%d default=%v", ret, defaultSym.Meta.DefaultMode)
	}
	if len(defaultObs.Snapshot().BitEvidence()) == 0 {
		t.Fatal("structurally defined default mode produced no evidence")
	}
}

// TestBitLLRAdditivity pins the accumulation semantics at the interface:
// agreeing signed evidence grows in magnitude, opposing evidence cancels.
func TestBitLLRAdditivity(t *testing.T) {
	a := BitLLRs([]float64{0.1, 2.0}, nil)[0]
	b := BitLLRs([]float64{0.3, 1.7}, nil)[0]
	c := BitLLRs([]float64{1.8, 0.1}, nil)[0]
	if a <= 0 || b <= 0 {
		t.Fatalf("agreeing observations not positive: %.2f, %.2f", a, b)
	}
	if c >= 0 {
		t.Fatalf("opposing observation not negative: %.2f", c)
	}
	if math.Abs(a+b) <= math.Abs(a) {
		t.Errorf("agreeing evidence did not grow: %.2f + %.2f", a, b)
	}
	if math.Abs(a+c) >= math.Abs(a) {
		t.Errorf("opposing evidence did not cancel: %.2f + %.2f", a, c)
	}
}
