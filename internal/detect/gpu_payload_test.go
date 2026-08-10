//go:build !js

package detect

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"math/bits"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

// deinterleavePermutation builds the resident permutation table and reads it
// back. Only a test wants the table itself: the chain applies it where it lies.
func (resident *gpuResidentBinarizer) deinterleavePermutation(
	params [gpuPayloadParamWords * 4]byte,
	length int,
) ([]uint32, error) {
	resident.mu.Lock()
	defer resident.mu.Unlock()
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, err
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.payloadParams, 0, params[:]); err != nil {
		return nil, err
	}
	if err := recorder.Fill(
		resident.payloadPermutationCache, 0, gpuPayloadPermutationCacheWords*4, 0,
	); err != nil {
		return nil, err
	}
	if err := recorder.Barrier(resident.payloadParams, resident.payloadPermutationCache); err != nil {
		return nil, err
	}
	if err := recorder.Dispatch(
		resident.payloadPermuteKernel,
		resident.payloadPermuteBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return nil, err
	}
	if err := recorder.Barrier(resident.payloadPermutation); err != nil {
		return nil, err
	}
	out := make([]byte, length*4)
	if err := recorder.Download(resident.payloadPermutation, 0, out); err != nil {
		return nil, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, err
	}
	// The table just built is not the one any later correction expects.
	resident.permutationCacheDirty = true
	permutation := make([]uint32, length)
	for at := range permutation {
		permutation[at] = binary.LittleEndian.Uint32(out[at*4:])
	}
	return permutation, nil
}

// gpuPayloadTestModule is the pixels per module the payload fixtures render at.
// It is comfortably above the sampler's small-module regime, so the grid the
// two arms share is a clean read of the encoder's own modules rather than a
// measurement of the sampler.
const gpuPayloadTestModule = 8
const gpuPayloadTestMargin = 2 * gpuPayloadTestModule

// gpuPayloadFixture is a rendered symbol placed in a frame at a whole number of
// pixels per module, with the finder centres the sampler's transform needs.
type gpuPayloadFixture struct {
	frame   *core.Bitmap
	modules *core.Bitmap
	palette []byte
	side    image.Point
	quad    [4]core.PointF
	colors  int

	// defaulted is the encoder's own condition for omitting explicit metadata,
	// restated so a fixture says which ladder it exercises instead of the test
	// discovering it from the arm under test.
	defaulted bool
}

// gpuPayloadVariant is the wire variant a colour mode is legal under. ISO
// admits four and eight colours and the host observation rejects any other mode
// on that variant outright, so a fixture above eight has to be walked as high
// colour or neither arm will read it.
func gpuPayloadVariant(colors int) wire.Variant {
	if colors > 8 {
		return wire.ISOHighColor
	}
	return wire.ISO23634
}

func gpuPayloadRender(t *testing.T, colors, eccLevel int, payload []byte) gpuPayloadFixture {
	t.Helper()
	rendered, err := encode.Render(encode.Config{
		Colors: colors, ModuleSize: 1, ECCLevel: eccLevel, SymbolNumber: 1,
	}, payload)
	if err != nil {
		t.Fatalf("render %d-colour symbol: %v", colors, err)
	}
	side := rendered.SideSize
	width := side.X*gpuPayloadTestModule + 2*gpuPayloadTestMargin
	height := side.Y*gpuPayloadTestModule + 2*gpuPayloadTestMargin
	frame := core.NewBitmap(width, height, 4)
	modules := core.NewBitmap(side.X, side.Y, 4)
	for at := range width * height {
		copy(frame.Pix[at*4:], []byte{255, 255, 255, 255})
	}
	for y := range side.Y {
		for x := range side.X {
			rgb := rendered.Palette[int(rendered.Matrix[y*side.X+x])*3:]
			moduleAt := modules.Offset(x, y)
			copy(modules.Pix[moduleAt:], rgb[:3])
			modules.Pix[moduleAt+3] = 255
			for py := range gpuPayloadTestModule {
				row := (gpuPayloadTestMargin + y*gpuPayloadTestModule + py) * width
				for px := range gpuPayloadTestModule {
					at := (row + gpuPayloadTestMargin + x*gpuPayloadTestModule + px) * 4
					copy(frame.Pix[at:], rgb[:3])
					frame.Pix[at+3] = 255
				}
			}
		}
	}
	centre := func(mx, my float64) core.PointF {
		return core.Pt(
			float64(gpuPayloadTestMargin)+mx*gpuPayloadTestModule,
			float64(gpuPayloadTestMargin)+my*gpuPayloadTestModule,
		)
	}
	return gpuPayloadFixture{
		frame:     frame,
		modules:   modules,
		palette:   append([]byte(nil), rendered.Palette...),
		side:      side,
		colors:    colors,
		defaulted: colors == 8 && (eccLevel == 0 || eccLevel == spec.DefaultECCLevel),
		quad: [4]core.PointF{
			centre(3.5, 3.5),
			centre(float64(side.X)-3.5, 3.5),
			centre(float64(side.X)-3.5, float64(side.Y)-3.5),
			centre(3.5, float64(side.Y)-3.5),
		},
	}
}

// gpuPayloadAmbiguousModules moves a spread of data modules just across their
// last-bit colour boundary. The host soft-path gate establishes this pattern as
// a hard-correction failure with recoverable low-margin evidence; carrying the
// same pattern here makes the device comparison exercise the resident retry.
func gpuPayloadAmbiguousModules(t *testing.T, fixture *gpuPayloadFixture) int {
	t.Helper()
	obs, result := decode.ObservePrimary(fixture.modules, &core.DecodedSymbol{})
	if result != core.Success || obs == nil {
		t.Fatalf("observe clean modules for soft retry: %d", result)
	}
	snapshot := obs.Snapshot()
	if snapshot == nil {
		t.Fatal("snapshot clean modules for soft retry")
	}
	corrupted := 0
	for module, reserved := range snapshot.DataMap {
		if reserved != 0 || module%3 != 0 {
			continue
		}
		x, y := module%fixture.side.X, module/fixture.side.X
		moduleAt := fixture.modules.Offset(x, y)
		original := nearestFixtureColor(
			fixture.modules.Pix[moduleAt:moduleAt+3], fixture.palette, fixture.colors,
		)
		if original < 2 {
			continue
		}
		target := original ^ 1
		for py := range gpuPayloadTestModule {
			for px := range gpuPayloadTestModule {
				at := fixture.frame.Offset(
					gpuPayloadTestMargin+x*gpuPayloadTestModule+px,
					gpuPayloadTestMargin+y*gpuPayloadTestModule+py,
				)
				for channel := range 3 {
					from := float64(fixture.frame.Pix[at+channel])
					to := float64(fixture.palette[target*3+channel])
					fixture.frame.Pix[at+channel] = byte(from + (to-from)*0.52 + 0.5)
				}
			}
		}
		corrupted++
	}
	return corrupted
}

func nearestFixtureColor(rgb, palette []byte, colors int) int {
	best, index := math.MaxInt, 0
	for color := range colors {
		distance := 0
		for channel := range 3 {
			delta := int(rgb[channel]) - int(palette[color*3+channel])
			distance += delta * delta
		}
		if distance < best {
			best, index = distance, color
		}
	}
	return index
}

func TestGPUPayloadSoftFixtureHasAmbiguousSpread(t *testing.T) {
	fixture := gpuPayloadRender(t, 8, 10, bytes.Repeat([]byte("device soft retry "), 4))
	if corrupted := gpuPayloadAmbiguousModules(t, &fixture); corrupted < 40 {
		t.Fatalf("soft retry fixture corrupted only %d modules", corrupted)
	}
}

// TestGPUPayloadShapeAdmitsLargestColourDepth guards the allocation bound at
// the place that used to retain the eight-colour limit. High-colour symbols
// need up to eight bits per module even though the module map still has one
// entry per module.
func TestGPUPayloadShapeAdmitsLargestColourDepth(t *testing.T) {
	const dataModules = 19980
	matrix := &core.Bitmap{Width: gpuSampleMaxSide, Height: gpuSampleMaxSide}
	symbol := &core.DecodedSymbol{
		WireVariant: wire.ISOHighColor,
		SideSize:    image.Pt(gpuSampleMaxSide, gpuSampleMaxSide),
		Meta: core.Metadata{
			NC:  7,
			ECL: image.Pt(3, 6),
		},
		Palette: make([]byte, gpuPayloadMaxColors*3*spec.PaletteCopies(gpuPayloadMaxColors)),
	}
	shape, err := gpuPayloadShapeOf(core.PayloadRequest{
		Matrix:            matrix,
		Symbol:            symbol,
		DataModules:       dataModules,
		NormalizedPalette: make([]float64, gpuPayloadMaxColors*4*spec.PaletteCopies(gpuPayloadMaxColors)),
		PaletteThresholds: make([]float64, 3*spec.ColorPaletteNumber),
	})
	if err != nil {
		t.Fatalf("largest colour depth declined: %v", err)
	}
	if shape.gross <= gpuPayloadMapWords*3 {
		t.Fatalf("fixture gross length %d does not cross the former three-bit bound", shape.gross)
	}
}

// TestGPUPayloadControlMatchesHardBlockSplit pins the resident control words to
// the one host definition of the codeword split. The data-map shader writes the
// same fields after counting its map, so a future metadata-to-payload fusion can
// stop supplying them without changing what the corrector consumes.
func TestGPUPayloadControlMatchesHardBlockSplit(t *testing.T) {
	const dataModules = 1400
	for _, colors := range []int{4, 8, 16, 32, 64, 128, 256} {
		for _, ecl := range []image.Point{image.Pt(3, 4), image.Pt(7, 11)} {
			copies := spec.PaletteCopies(colors)
			symbol := &core.DecodedSymbol{
				WireVariant: wire.ISOHighColor,
				SideSize:    image.Pt(41, 41),
				Meta: core.Metadata{
					NC:  bits.Len(uint(colors)) - 2,
					ECL: ecl,
				},
				Palette: make([]byte, colors*3*copies),
			}
			shape, err := gpuPayloadShapeOf(core.PayloadRequest{
				Matrix:            &core.Bitmap{Width: 41, Height: 41},
				Symbol:            symbol,
				DataModules:       dataModules,
				NormalizedPalette: make([]float64, colors*4*copies),
				PaletteThresholds: make([]float64, 3*spec.ColorPaletteNumber),
			})
			if err != nil {
				t.Fatalf("colors %d ecl %v: shape: %v", colors, ecl, err)
			}
			word := func(index int) int {
				return int(binary.LittleEndian.Uint32(shape.params[index*4:]))
			}
			layout := ecc.HardBlockSplit(dataModules*(bits.Len(uint(colors))-1), ecl.X, ecl.Y)
			if word(gpuPayloadParamDataModules) != dataModules ||
				word(gpuPayloadParamGrossBits) != layout.Pg ||
				word(gpuPayloadParamNetBits) != layout.Pn ||
				word(gpuPayloadParamWC) != ecl.X || word(gpuPayloadParamWR) != ecl.Y {
				t.Fatalf("colors %d ecl %v: resident control disagrees with split %+v", colors, ecl, layout)
			}
			if shape.ldpc.length != layout.GrossSub || shape.ldpc.net != layout.NetSub ||
				shape.ldpc.blocks != layout.Blocks {
				t.Fatalf("colors %d ecl %v: correction plan %+v disagrees with split %+v",
					colors, ecl, shape.ldpc, layout)
			}
			if len(shape.ldpc.rows) != 0 || len(shape.ldpc.tailRows) != 0 {
				t.Fatalf("colors %d ecl %v: host materialized %d+%d payload row words",
					colors, ecl, len(shape.ldpc.rows), len(shape.ldpc.tailRows))
			}
			if !layout.Uniform && shape.ldpc.tailLength != layout.TrailingGrossSub() {
				t.Fatalf("colors %d ecl %v: tail %d, want %d",
					colors, ecl, shape.ldpc.tailLength, layout.TrailingGrossSub())
			}
		}
	}
}

// buildLDPCMatrix records the resident matrix builders and returns their sparse
// rows plus the control they resolved. It is a target-adapter parity seam, not
// part of production readback.
func (resident *gpuResidentBinarizer) buildLDPCMatrix(
	layout ecc.HardBlockLayout,
	variant wire.Variant,
) ([]byte, []byte, error) {
	resident.mu.Lock()
	defer resident.mu.Unlock()

	var payload [gpuPayloadParamWords * 4]byte
	put := func(index, value int) {
		binary.LittleEndian.PutUint32(payload[index*4:], uint32(value))
	}
	put(gpuPayloadParamWC, layout.WC)
	put(gpuPayloadParamWR, layout.WR)
	if !variant.UsesISO23634Base() {
		put(gpuPayloadParamGenerator, gpuPayloadGeneratorLCG)
	}
	plan := gpuLDPCPlan{
		length: layout.GrossSub,
		net:    layout.NetSub,
		blocks: layout.Blocks,
	}
	if !layout.Uniform {
		plan.tailLength = layout.TrailingGrossSub()
		plan.tailNet = plan.tailLength * (layout.WR - layout.WC) / layout.WR
	}
	params := gpuLDPCParams(plan)

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, nil, err
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.payloadParams, 0, payload[:]); err != nil {
		return nil, nil, err
	}
	if err := recorder.Update(resident.ldpcParams, 0, params[:]); err != nil {
		return nil, nil, err
	}
	if err := recorder.Fill(resident.ldpcMatrixCache, 0, gpuLDPCMatrixCacheWords*4, 0); err != nil {
		return nil, nil, err
	}
	if err := recorder.Barrier(resident.payloadParams, resident.ldpcParams, resident.ldpcMatrixCache); err != nil {
		return nil, nil, err
	}
	if err := recorder.Dispatch(
		resident.ldpcMatrixKernel,
		resident.ldpcMatrixBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return nil, nil, err
	}
	if err := recorder.Barrier(
		resident.ldpcRows, resident.ldpcParams,
		resident.ldpcMatrixScratch, resident.ldpcMatrixCache,
	); err != nil {
		return nil, nil, err
	}
	if err := recorder.Dispatch(
		resident.ldpcTailMatrixKernel,
		resident.ldpcTailMatrixBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return nil, nil, err
	}
	if err := recorder.Barrier(resident.ldpcRows, resident.ldpcParams); err != nil {
		return nil, nil, err
	}
	rows := make([]byte, 2*gpuLDPCRowWords*4)
	control := make([]byte, gpuLDPCParamWords*4)
	if err := recorder.Download(resident.ldpcRows, 0, rows); err != nil {
		return nil, nil, err
	}
	if err := recorder.Download(resident.ldpcParams, 0, control); err != nil {
		return nil, nil, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, nil, err
	}
	resident.ldpcMatrixCacheDirty = true
	return rows, control, nil
}

// TestGPULDPCMatrixMatchesHost is the real-adapter gate for the selected matrix
// builder. Hard LDPC can accept a plausible wrong codeword, so checking only a
// decoded fixture is weaker than comparing every emitted sparse row and rank.
func TestGPULDPCMatrixMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU matrix test device: %v", err)
		}
	})

	findLayout := func(bitsPerModule int, uniform bool) ecc.HardBlockLayout {
		for modules := 64; modules < 3000; modules++ {
			layout := ecc.HardBlockSplit(modules*bitsPerModule, 3, 4)
			if layout.Blocks > 0 && layout.GrossSub <= gpuLDPCMaxSub && layout.Uniform == uniform {
				return layout
			}
		}
		t.Fatalf("no uniform=%t layout for %d bits per module", uniform, bitsPerModule)
		return ecc.HardBlockLayout{}
	}
	word := func(control []byte, index int) int {
		return int(binary.LittleEndian.Uint32(control[index*4:]))
	}
	checkRows := func(name string, rows []byte, base int, gotHeight, gotDegree, gotRank int,
		want ecc.ParityRowLayout) {
		t.Helper()
		if gotHeight != want.Height || gotDegree != want.Degree || gotRank != want.Rank {
			t.Fatalf("%s control = height %d degree %d rank %d, want %d %d %d",
				name, gotHeight, gotDegree, gotRank, want.Height, want.Degree, want.Rank)
		}
		for at, column := range want.Rows {
			got := binary.LittleEndian.Uint32(rows[(base+at)*4:])
			if got != column {
				t.Fatalf("%s row word %d = %d, want %d", name, at, got, column)
			}
		}
	}

	for _, colors := range []int{4, 8, 16, 32, 64, 128, 256} {
		bitsPerModule := bits.Len(uint(colors)) - 1
		for _, uniform := range []bool{true, false} {
			layout := findLayout(bitsPerModule, uniform)
			for _, variant := range []wire.Variant{wire.ISOHighColor, wire.CurrentC} {
				name := fmt.Sprintf("colors=%d/uniform=%t/variant=%d", colors, uniform, variant)
				t.Run(name, func(t *testing.T) {
					rows, control, err := resident.buildLDPCMatrix(layout, variant)
					if err != nil {
						t.Fatalf("build matrix: %v", err)
					}
					want, ok := ecc.ParityRows(layout.WC, layout.WR, layout.GrossSub, variant)
					if !ok {
						t.Fatal("host regular matrix unavailable")
					}
					checkRows("regular", rows, 0,
						word(control, gpuLDPCParamHeight),
						word(control, gpuLDPCParamRowDegree),
						word(control, gpuLDPCParamRank), want)
					if layout.Uniform {
						return
					}
					tail, ok := ecc.ParityRows(
						layout.WC, layout.WR, layout.TrailingGrossSub(), variant)
					if !ok {
						t.Fatal("host trailing matrix unavailable")
					}
					checkRows("tail", rows, gpuLDPCRowWords,
						word(control, gpuLDPCParamTailHeight),
						word(control, gpuLDPCParamTailRowDegree),
						word(control, gpuLDPCParamTailRank), tail)
				})
			}
		}
	}
}

// TestGPUDeinterleavePermutationMatchesHost pins the resident permutation table
// against the host's for both of the format's generators.
//
// The table decides where every codeword bit lands, so a generator that drifts
// by one value scrambles the whole codeword rather than a corner of it. The
// C-family generator is the one worth testing here: its 64-bit state is carried
// as two 32-bit halves on a device with no 64-bit integer, and its index scaling
// runs through the same f32 rounding the reference's does.
func TestGPUDeinterleavePermutationMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU permutation test device: %v", err)
		}
	})

	for _, test := range []struct {
		name      string
		variant   wire.Variant
		generator uint32
	}{
		{name: "ISO", variant: wire.ISO23634},
		{name: "C family", variant: wire.CurrentC, generator: gpuPayloadGeneratorLCG},
	} {
		t.Run(test.name, func(t *testing.T) {
			for _, length := range []int{1, 2, 63, 4097} {
				var params [gpuPayloadParamWords * 4]byte
				binary.LittleEndian.PutUint32(
					params[gpuPayloadParamGrossBits*4:], uint32(length))
				binary.LittleEndian.PutUint32(
					params[gpuPayloadParamGenerator*4:], test.generator)
				got, err := resident.deinterleavePermutation(params, length)
				if err != nil {
					t.Fatalf("length %d: device permutation: %v", length, err)
				}
				want := ecc.DeinterleavePermutation(length, test.variant)
				for at := range want {
					if got[at] != want[at] {
						t.Fatalf("length %d: device sends element %d to %d, host to %d",
							length, at, got[at], want[at])
					}
				}
			}
		})
	}
}

// TestGPUPayloadChainMatchesHost holds the device payload chain to the host
// chain over the same sampled grid: the data map, palette classification,
// unmasking, bit packing, deinterleaving, hard correction and its resident soft
// retry together.
//
// Hard LDPC carries no payload integrity check, so a chain that classifies one
// module differently can still return a plausible payload with no error. The
// comparison is therefore the decoded bytes in full, on a symbol whose payload
// is known, and both arms must reach success rather than agreeing on failure.
func TestGPUPayloadChainMatchesHost(t *testing.T) {
	payload := bytes.Repeat([]byte("device payload chain "), 4)
	// The higher modes classify by absolute palette distance against two
	// embedded copies rather than by normalized direction against four, and fold
	// the nearest spatial corner onto a copy. That is a second classifier, and
	// the byte comparison over a known payload is the only thing that catches it
	// disagreeing with the host by one module.
	fixtures := map[string]gpuPayloadFixture{
		"8 colour":   gpuPayloadRender(t, 8, 10, payload),
		"4 colour":   gpuPayloadRender(t, 4, 6, payload),
		"16 colour":  gpuPayloadRender(t, 16, 6, payload),
		"32 colour":  gpuPayloadRender(t, 32, 6, payload),
		"64 colour":  gpuPayloadRender(t, 64, 6, payload),
		"128 colour": gpuPayloadRender(t, 128, 6, payload),
		"256 colour": gpuPayloadRender(t, 256, 6, payload),
		// A symbol carrying no explicit metadata at all, which the device walk
		// used to decline outright and which now runs the whole chain.
		"default mode": gpuPayloadRender(t, 8, spec.DefaultECCLevel, payload),
	}
	softFixture := gpuPayloadRender(t, 8, 10, payload)
	if corrupted := gpuPayloadAmbiguousModules(t, &softFixture); corrupted < 40 {
		t.Fatalf("soft retry fixture corrupted only %d modules", corrupted)
	}
	fixtures["soft retry"] = softFixture
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
		t.Fatalf("allocate GPU payload test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, maxWidth, maxHeight)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		// The resident binarizer caches binding sets against the input buffer,
		// so it has to release them before the buffer can be closed.
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU payload test input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU payload test device: %v", err)
		}
	})

	for name, fixture := range fixtures {
		t.Run(name, func(t *testing.T) {
			width, height := fixture.frame.Width, fixture.frame.Height
			frame := make([]byte, frameBytes)
			copy(frame, fixture.frame.Pix)
			if err := input.Upload(frame); err != nil {
				t.Fatalf("upload GPU payload test input: %v", err)
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
			// The device chain reads the resident grid; the host chain it is
			// being compared against reads modules, so they come across here.
			if !resident.MaterializeGrid(matrix) {
				t.Fatal("could not materialize the sampled grid for the host chain")
			}

			hostSymbol := &core.DecodedSymbol{WireVariant: gpuPayloadVariant(fixture.colors)}
			hostObs, ret := decode.ObservePrimary(matrix, hostSymbol)
			if ret != core.Success || hostObs == nil {
				t.Fatalf("host observation of the sampled grid failed: %d", ret)
			}
			if hostSymbol.Meta.DefaultMode != fixture.defaulted {
				t.Fatalf("host default mode %v, fixture says %v",
					hostSymbol.Meta.DefaultMode, fixture.defaulted)
			}
			if !hostObs.AdmitPayloadCorrection() {
				t.Fatal("host admission rejected the clean sampled grid")
			}
			if got := hostObs.CorrectPayload(); got != core.Success {
				t.Fatalf("host payload correction failed: %d", got)
			}

			deviceSymbol := &core.DecodedSymbol{WireVariant: gpuPayloadVariant(fixture.colors)}
			deviceObs, ret, handled := decode.ObservePrimaryOnDevice(
				resident, matrix, deviceSymbol, nil)
			if !handled || ret != core.Success || deviceObs == nil {
				t.Fatalf("device-arm observation of the sampled grid failed: %d", ret)
			}
			deviceObs.UseDevice(resident)
			if !deviceObs.AdmitPayloadCorrection() {
				t.Fatal("device admission rejected the clean sampled grid")
			}
			// A completed cache update proves the device chain ran: a decline to
			// the host leaves this dirty and would make the comparison vacuous.
			resident.permutationCacheDirty = true
			if got := deviceObs.CorrectPayload(); got != core.Success {
				t.Fatalf("device payload correction failed: %d", got)
			}
			if resident.permutationCacheDirty {
				t.Fatal("the device payload chain declined; the comparison would be vacuous")
			}

			if !bytes.Equal(deviceSymbol.Data, hostSymbol.Data) {
				differing := 0
				for at := range min(len(deviceSymbol.Data), len(hostSymbol.Data)) {
					if deviceSymbol.Data[at] != hostSymbol.Data[at] {
						differing++
					}
				}
				t.Fatalf("device stream is %d bits, host %d, %d differing",
					len(deviceSymbol.Data), len(hostSymbol.Data), differing)
			}
			if deviceSymbol.Meta.DockedPosition != hostSymbol.Meta.DockedPosition {
				t.Errorf("device docked position %d, host %d",
					deviceSymbol.Meta.DockedPosition, hostSymbol.Meta.DockedPosition)
			}
		})
	}
}
