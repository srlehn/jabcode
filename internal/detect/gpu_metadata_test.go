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

// hostMetadataWalk runs the host's metadata strip over a sampled grid and
// reports what the device stages report, so the two can be compared without the
// test reimplementing either side.
func hostMetadataWalk(t *testing.T, matrix *core.Bitmap) gpuMetadataWalk {
	t.Helper()
	symbol := &core.DecodedSymbol{SideSize: image.Pt(matrix.Width, matrix.Height)}
	dataMap := make([]byte, matrix.Width*matrix.Height)
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	count := 0
	ret, syndromeOK := decode.DecodePrimaryMetadataPartI(matrix, symbol, dataMap, &count, &x, &y)
	if ret == decode.MetadataFailed {
		return gpuMetadataWalk{Defaulted: true, ModuleCount: count}
	}
	if ret != core.Success {
		t.Fatalf("host Part I failed on the sampled grid: %d", ret)
	}
	if got := decode.ReadColorPaletteInPrimary(matrix, symbol, dataMap, &count, &x, &y); got < 0 {
		t.Fatalf("host palette read failed on the sampled grid: %d", got)
	}
	colors := 1 << (symbol.Meta.NC + 1)
	copies := spec.PaletteCopies(colors)
	walk := gpuMetadataWalk{
		NC: symbol.Meta.NC, Colors: colors, SyndromeOK: syndromeOK, ModuleCount: count,
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
		for i := range want.Palette {
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
	fixtures := map[string]gpuPayloadFixture{
		"8 colour": gpuPayloadRender(t, 8, 10, payload),
		"4 colour": gpuPayloadRender(t, 4, 6, payload),
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

			want := hostMetadataWalk(t, matrix)
			got, err := resident.WalkMetadata(fixture.side, wire.ISO23634)
			if err != nil {
				t.Fatalf("device metadata walk: %v", err)
			}
			if got.Defaulted != want.Defaulted {
				t.Fatalf("device defaulted=%t, host defaulted=%t", got.Defaulted, want.Defaulted)
			}
			if want.Defaulted {
				return
			}
			if got.Unsupported {
				t.Fatalf("device declined a %d-colour symbol the host read", want.Colors)
			}
			if got.NC != want.NC {
				t.Fatalf("device NC=%d, host NC=%d", got.NC, want.NC)
			}
			if got.SyndromeOK != want.SyndromeOK {
				t.Errorf("device syndrome ok=%t, host ok=%t", got.SyndromeOK, want.SyndromeOK)
			}
			comparePaletteWalk(t, got, want)
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
		return matrix
	}

	clean := hostMetadataWalk(t, sample(fixture.frame.Pix, "clean"))
	if clean.Defaulted {
		t.Fatal("the clean fixture already defaults, so there is no colour mode to recover")
	}

	matrix := sample(castPartIBlackModules(t, fixture).Pix, "cast")
	if plainPartIResolves(matrix) {
		t.Fatal("the cast left Part I's plain classification working, so the retry arm never ran")
	}
	want := hostMetadataWalk(t, matrix)
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
	if got.NC != want.NC || got.SyndromeOK != want.SyndromeOK {
		t.Errorf("device NC=%d syndrome=%t, host NC=%d syndrome=%t",
			got.NC, got.SyndromeOK, want.NC, want.SyndromeOK)
	}
	// Agreeing on a wrong answer would satisfy the comparison above, so the
	// recovered mode is held to the one the clean fixture reads.
	if got.NC != clean.NC {
		t.Errorf("the retry recovered NC=%d, but the clean fixture reads NC=%d", got.NC, clean.NC)
	}
}
