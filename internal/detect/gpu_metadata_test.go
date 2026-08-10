//go:build !js

package detect

import (
	"bytes"
	"image"
	"math"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestGPUMetadataLDPCRowBound(t *testing.T) {
	for _, variant := range []wire.Variant{wire.ISO23634, wire.CurrentC} {
		for _, build := range []func(wire.Variant) (gpuLDPCPlan, error){
			gpuMetadataPartIPlan,
			gpuMetadataPartIIPlan,
		} {
			plan, err := build(variant)
			if err != nil {
				t.Fatalf("variant %d: metadata plan: %v", variant, err)
			}
			if len(plan.rows) > gpuMetadataLDPCRowWords || len(plan.tailRows) != 0 {
				t.Fatalf("variant %d: metadata rows %d+%d exceed %d words",
					variant, len(plan.rows), len(plan.tailRows), gpuMetadataLDPCRowWords)
			}
		}
	}
}

// hostMetadataWalk runs the host's metadata strip over a sampled grid and
// reports what the device stages report, so the two can be compared without the
// test reimplementing either side.
func hostMetadataWalk(t *testing.T, matrix *core.Bitmap, variant wire.Variant) gpuMetadataWalk {
	t.Helper()
	symbol := &core.DecodedSymbol{
		SideSize:    image.Pt(matrix.Width, matrix.Height),
		WireVariant: variant,
	}
	dataMap := make([]byte, matrix.Width*matrix.Height)
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	count := 0
	ret, syndromeOK := decode.DecodePrimaryMetadataPartI(matrix, symbol, dataMap, &count, &x, &y)
	defaulted := ret == decode.MetadataFailed
	if defaulted {
		// The host's own ladder: default metadata, the walk restarted at the
		// strip's beginning, and no Part II. The device does the same, so the
		// comparison has to follow it rather than stop here.
		x, y = spec.PrimaryMetadataX, spec.PrimaryMetadataY
		count = 0
		clear(dataMap)
		decode.LoadDefaultPrimaryMetadata(matrix, symbol)
	} else if ret != core.Success {
		t.Fatalf("host Part I failed on the sampled grid: %d", ret)
	}
	if got := decode.ReadColorPaletteInPrimary(matrix, symbol, dataMap, &count, &x, &y); got < 0 {
		t.Fatalf("host palette read failed on the sampled grid: %d", got)
	}
	colors := 1 << (symbol.Meta.NC + 1)
	copies := spec.PaletteCopies(colors)
	walk := gpuMetadataWalk{
		NC: symbol.Meta.NC, Colors: colors, PartISyndromeOK: syndromeOK,
		Defaulted:  defaulted,
		Palette:    symbol.Palette,
		Normalized: make([]float64, colors*4*copies),
		Thresholds: make([]float64, 3*spec.ColorPaletteNumber),
	}
	decode.NormalizeColorPalette(symbol, walk.Normalized, colors)
	for copy := range copies {
		threshold := decode.PaletteThreshold(symbol.Palette[colors*3*copy:], colors)
		walk.Thresholds[copy*3+0] = threshold[0]
		walk.Thresholds[copy*3+1] = threshold[1]
		walk.Thresholds[copy*3+2] = threshold[2]
	}
	if defaulted {
		// A default-mode symbol has no Part II, and the device reports no shape
		// for one either: the caller takes the format's constants.
		walk.ModuleCount = count
		walk.PartISyndromeOK = false
		return walk
	}
	ret, partII := decode.DecodePrimaryMetadataPartII(
		matrix, symbol, dataMap, walk.Normalized, walk.Thresholds, &count, &x, &y)
	walk.ModuleCount = count
	walk.PartIISyndromeOK = partII
	walk.Rejected = ret != core.Success
	walk.SideVersion = symbol.Meta.SideVersion
	walk.ECL = symbol.Meta.ECL
	walk.MaskType = symbol.Meta.MaskType
	return walk
}

// comparePaletteWalk holds the device's palette and everything derived from it
// to the host's. The palette bytes and the thresholds are integer arithmetic on
// both sides and must be exact; only the normalized entries are a float
// division, and the device does it in f32 where the host has f64.
func comparePaletteWalk(t *testing.T, got, want gpuMetadataWalk) {
	t.Helper()
	if got.Colors != want.Colors {
		t.Fatalf("device read a %d-colour palette, host %d", got.Colors, want.Colors)
	}
	if got.ModuleCount != want.ModuleCount {
		t.Errorf("device consumed %d metadata modules, host %d", got.ModuleCount, want.ModuleCount)
	}
	if !bytes.Equal(got.Palette, want.Palette) {
		for i := range min(len(want.Palette), len(got.Palette)) {
			if got.Palette[i] != want.Palette[i] {
				t.Fatalf("palette copy %d entry %d channel %d is %d, host %d",
					i/(want.Colors*3), (i%(want.Colors*3))/3, i%3,
					got.Palette[i], want.Palette[i])
			}
		}
		t.Fatalf("device palette is %d bytes, host %d", len(got.Palette), len(want.Palette))
	}
	for i := range want.Thresholds {
		if got.Thresholds[i] != want.Thresholds[i] {
			t.Errorf("threshold copy %d channel %d is %g, host %g",
				i/3, i%3, got.Thresholds[i], want.Thresholds[i])
		}
	}
	// Above eight colours the device derives no normalized palette, because the
	// classifier for those modes ranks absolute distance against the palette
	// bytes and would never read one. Asserting that it is absent is the check;
	// deriving it anyway to compare would be testing code the decode never runs.
	if want.Colors > 8 {
		if len(got.Normalized) != 0 {
			t.Errorf("device derived %d normalized entries for a %d-colour mode that classifies by absolute distance",
				len(got.Normalized), want.Colors)
		}
		return
	}
	const normalizedTolerance = 1e-6
	worst := 0.0
	for i := range want.Normalized {
		if delta := math.Abs(got.Normalized[i] - want.Normalized[i]); delta > worst {
			worst = delta
		}
	}
	if worst > normalizedTolerance {
		t.Errorf("normalized palette differs from the host by up to %g", worst)
	}
	t.Logf("%d-colour palette exact, normalized within %g", want.Colors, worst)
}

// compareMetadataWalk holds the whole device interpretation to the host's. The
// symbol shape is compared field by field rather than through a single verdict:
// a wrong side version or mask reference produces a different codeword, and hard
// LDPC has nothing underneath it to notice.
func compareMetadataWalk(t *testing.T, got, want gpuMetadataWalk) {
	t.Helper()
	if got.PartISyndromeOK != want.PartISyndromeOK {
		t.Errorf("device Part I syndrome ok=%t, host ok=%t",
			got.PartISyndromeOK, want.PartISyndromeOK)
	}
	comparePaletteWalk(t, got, want)
	if got.Rejected != want.Rejected {
		t.Fatalf("device rejected=%t, host rejected=%t", got.Rejected, want.Rejected)
	}
	if got.SideVersion != want.SideVersion {
		t.Errorf("device side version %v, host %v", got.SideVersion, want.SideVersion)
	}
	if got.ECL != want.ECL {
		t.Errorf("device ECC weights %v, host %v", got.ECL, want.ECL)
	}
	if got.MaskType != want.MaskType {
		t.Errorf("device mask %d, host %d", got.MaskType, want.MaskType)
	}
	if got.PartIISyndromeOK != want.PartIISyndromeOK {
		t.Errorf("device Part II syndrome ok=%t, host ok=%t",
			got.PartIISyndromeOK, want.PartIISyndromeOK)
	}
	t.Logf("version %v ECC %v mask %d, %d metadata modules",
		got.SideVersion, got.ECL, got.MaskType, got.ModuleCount)
}

// TestGPUMetadataWalkMatchesHost holds the device metadata walk to the host's
// over grids the device itself sampled.
//
// Part I decides the colour mode, and everything after it - how many palette
// modules there are, which entry each carries, the classifier the payload is
// read with - is built on that answer, so a divergence anywhere here is not a
// degraded read but a different symbol. The comparison is therefore the whole
// walk at once: colour mode, default fallback, parity verdict, the palette
// bytes, the normalized entries and the thresholds.
func TestGPUMetadataWalkMatchesHost(t *testing.T) {
	payload := bytes.Repeat([]byte("metadata part one "), 4)
	// The higher modes are here because they are a different walk, not a longer
	// one: two embedded palette copies instead of four, no colour taken from the
	// finder, no black threshold, and a classifier ranking absolute distance.
	// Each of those is a place the two arms can diverge silently, and hard LDPC
	// has no payload integrity check to catch it downstream.
	fixtures := map[string]gpuPayloadFixture{
		"8 colour":   gpuPayloadRender(t, 8, 10, payload),
		"4 colour":   gpuPayloadRender(t, 4, 6, payload),
		"16 colour":  gpuPayloadRender(t, 16, 6, payload),
		"32 colour":  gpuPayloadRender(t, 32, 6, payload),
		"64 colour":  gpuPayloadRender(t, 64, 6, payload),
		"128 colour": gpuPayloadRender(t, 128, 6, payload),
		"256 colour": gpuPayloadRender(t, 256, 6, payload),
		// Eight colours at the default ECC level carries no explicit metadata at
		// all, which is the ladder the device used to decline outright.
		"default mode": gpuPayloadRender(t, 8, spec.DefaultECCLevel, payload),
	}
	maxWidth, maxHeight := 0, 0
	for _, fixture := range fixtures {
		maxWidth = max(maxWidth, fixture.frame.Width)
		maxHeight = max(maxHeight, fixture.frame.Height)
	}

	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	frameBytes := uint64(maxWidth) * uint64(maxHeight) * 4
	input, err := device.NewBuffer(frameBytes)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU metadata test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, maxWidth, maxHeight)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU metadata test input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU metadata test device: %v", err)
		}
	})

	for name, fixture := range fixtures {
		t.Run(name, func(t *testing.T) {
			width, height := fixture.frame.Width, fixture.frame.Height
			frame := make([]byte, frameBytes)
			copy(frame, fixture.frame.Pix)
			if err := input.Upload(frame); err != nil {
				t.Fatalf("upload GPU metadata test input: %v", err)
			}
			if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
				t.Fatalf("resident GPU Binarize: %v", err)
			}
			pt := core.PerspectiveTransform(
				fixture.quad[0], fixture.quad[1], fixture.quad[2], fixture.quad[3], fixture.side)
			matrix, err := resident.SampleSymbol(width, height, pt, fixture.side, [3]core.PointF{})
			if err != nil {
				t.Fatalf("GPU SampleSymbol: %v", err)
			}
			if matrix == nil {
				t.Fatal("GPU sampler rejected the fixture geometry")
			}

			// The device walk reads the resident grid; the host comparison
			// needs the modules on this side, which is what materializing is
			// for. Order matters: materializing first would still be reading
			// the same sample, but taking the device answer first keeps the
			// comparison honest about what the device saw.
			// A decode rederives the normalized palette and the thresholds and
			// so never fetches the device's, but this comparison is about
			// exactly those, so it asks for the whole record.
			resident.metadataFetchDerived = true
			variant := gpuPayloadVariant(fixture.colors)
			got, err := resident.WalkMetadata(fixture.side, variant)
			if err != nil {
				t.Fatalf("device metadata walk: %v", err)
			}
			if !resident.MaterializeGrid(matrix) {
				t.Fatal("could not materialize the sampled grid for the host walk")
			}
			want := hostMetadataWalk(t, matrix, variant)
			if err != nil {
				t.Fatalf("device metadata walk: %v", err)
			}
			if got.Defaulted != want.Defaulted {
				t.Fatalf("device defaulted=%t, host defaulted=%t", got.Defaulted, want.Defaulted)
			}
			// Without this a fixture that quietly stopped defaulting would make
			// both arms agree on the explicit ladder and prove nothing about the
			// default one.
			if got.Defaulted != fixture.defaulted {
				t.Fatalf("fixture expects defaulted=%t, both arms read %t",
					fixture.defaulted, got.Defaulted)
			}
			if got.Unsupported {
				t.Fatalf("device declined a %d-colour symbol the host read", want.Colors)
			}
			if got.NC != want.NC {
				t.Fatalf("device NC=%d, host NC=%d", got.NC, want.NC)
			}
			compareMetadataWalk(t, got, want)
		})
	}
}

// plainPartIResolves reports whether Part I's ordinary classification already
// yields a legal colour pair. The reference-anchored retry runs only when it
// does not, and the host does not say which arm it took, so a test that means to
// exercise the retry has to establish that the plain arm failed first.
func plainPartIResolves(matrix *core.Bitmap) bool {
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	var colors [spec.PrimaryMetadataPart1ModuleNumber]byte
	for count := range colors {
		offset := (y*matrix.Width + x) * matrix.Channels
		colors[count] = decode.DecodeModuleNC(matrix.Pix[offset : offset+3])
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, count+1, &x, &y)
	}
	index := func(color byte) int {
		switch color {
		case 0:
			return 0
		case 3:
			return 1
		case 6:
			return 2
		}
		return 3
	}
	pair := func(first, second byte) bool {
		a, b := index(first), index(second)
		return a <= 2 && b <= 2 && a*3+b <= 7
	}
	return pair(colors[0], colors[1]) && pair(colors[2], colors[3])
}

// castPartIBlackModules rewrites every black Part I module of a rendered fixture
// to a dark red, which is what makes the reference-anchored retry the arm that
// answers.
//
// The obvious cast - lifting a channel across the frame - does not reach Part I
// at all, because the resident binarizer balances the frame before the sampler
// sees it and a uniform lift is exactly what balancing removes. A per-module
// substitution survives it: dark red fails the absolute black test, classifies
// as a value the Part I encoding has no colour for, and still sits nearest the
// black reference, so the retry recovers the module the plain arm lost.
func castPartIBlackModules(t *testing.T, fixture gpuPayloadFixture) *core.Bitmap {
	t.Helper()
	const margin = 2 * gpuPayloadTestModule
	frame := &core.Bitmap{
		Width: fixture.frame.Width, Height: fixture.frame.Height, Channels: 4,
		Pix: append([]byte(nil), fixture.frame.Pix...),
	}
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	cast := 0
	for count := range spec.PrimaryMetadataPart1ModuleNumber {
		at := (margin + y*gpuPayloadTestModule) * frame.Width
		at = (at + margin + x*gpuPayloadTestModule) * 4
		if decode.DecodeModuleNC(frame.Pix[at:at+3]) == 0 {
			for py := range gpuPayloadTestModule {
				row := (margin + y*gpuPayloadTestModule + py) * frame.Width
				for px := range gpuPayloadTestModule {
					offset := (row + margin + x*gpuPayloadTestModule + px) * 4
					copy(frame.Pix[offset:], []byte{100, 0, 0, 255})
				}
			}
			cast++
		}
		spec.NextMetadataModuleInPrimary(fixture.side.Y, fixture.side.X, count+1, &x, &y)
	}
	if cast == 0 {
		t.Fatal("no Part I module of the fixture is black, so nothing drives the retry")
	}
	return frame
}

// TestGPUMetadataPartIReferenceRetry drives the arm a clean fixture never
// reaches. Part I decides from absolute channel values, which a capture's
// colour cast defeats, and the host then re-classifies the same four modules
// against references derived from the symbol's own finder cores. The device has
// to take that arm on the same modules and recover the same colour mode, or a
// cast capture silently decodes as a different symbol.
func TestGPUMetadataPartIReferenceRetry(t *testing.T) {
	payload := bytes.Repeat([]byte("cast "), 8)
	fixture := gpuPayloadRender(t, 8, 10, payload)

	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	width, height := fixture.frame.Width, fixture.frame.Height
	frameBytes := uint64(width) * uint64(height) * 4
	input, err := device.NewBuffer(frameBytes)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU metadata cast input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
	})
	pt := core.PerspectiveTransform(
		fixture.quad[0], fixture.quad[1], fixture.quad[2], fixture.quad[3], fixture.side)

	sample := func(pixels []byte, what string) *core.Bitmap {
		t.Helper()
		if err := input.Upload(pixels); err != nil {
			t.Fatalf("upload the %s frame: %v", what, err)
		}
		if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
			t.Fatalf("resident GPU Binarize of the %s frame: %v", what, err)
		}
		matrix, err := resident.SampleSymbol(width, height, pt, fixture.side, [3]core.PointF{})
		if err != nil {
			t.Fatalf("GPU SampleSymbol of the %s frame: %v", what, err)
		}
		if matrix == nil {
			t.Fatalf("GPU sampler rejected the %s frame", what)
		}
		// The host comparisons below read modules, so this sample's grid comes
		// across before the next one overwrites it.
		if !resident.MaterializeGrid(matrix) {
			t.Fatalf("could not materialize the %s frame's sampled grid", what)
		}
		return matrix
	}

	clean := hostMetadataWalk(t, sample(fixture.frame.Pix, "clean"), wire.ISO23634)
	if clean.Defaulted {
		t.Fatal("the clean fixture already defaults, so there is no colour mode to recover")
	}

	matrix := sample(castPartIBlackModules(t, fixture).Pix, "cast")
	if plainPartIResolves(matrix) {
		t.Fatal("the cast left Part I's plain classification working, so the retry arm never ran")
	}
	want := hostMetadataWalk(t, matrix, wire.ISO23634)
	if want.Defaulted {
		t.Fatal("the cast drove the host to default metadata, so the retry arm never ran")
	}

	got, err := resident.WalkMetadata(fixture.side, wire.ISO23634)
	if err != nil {
		t.Fatalf("device metadata Part I: %v", err)
	}
	if got.Defaulted {
		t.Fatal("device fell back to default metadata where the host's retry resolved")
	}
	if got.NC != want.NC || got.PartISyndromeOK != want.PartISyndromeOK {
		t.Errorf("device NC=%d syndrome=%t, host NC=%d syndrome=%t",
			got.NC, got.PartISyndromeOK, want.NC, want.PartISyndromeOK)
	}
	// Agreeing on a wrong answer would satisfy the comparison above, so the
	// recovered mode is held to the one the clean fixture reads.
	if got.NC != clean.NC {
		t.Errorf("the retry recovered NC=%d, but the clean fixture reads NC=%d", got.NC, clean.NC)
	}
}
