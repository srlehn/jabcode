//go:build !js

package detect

import (
	"bytes"
	"encoding/binary"
	"image"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/encode"
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
	if err := recorder.Barrier(resident.payloadParams); err != nil {
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
	resident.permutationLength = 0
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

// gpuPayloadFixture is a rendered symbol placed in a frame at a whole number of
// pixels per module, with the finder centres the sampler's transform needs.
type gpuPayloadFixture struct {
	frame *core.Bitmap
	side  image.Point
	quad  [4]core.PointF
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
	const margin = 2 * gpuPayloadTestModule
	width := side.X*gpuPayloadTestModule + 2*margin
	height := side.Y*gpuPayloadTestModule + 2*margin
	frame := core.NewBitmap(width, height, 4)
	for at := range width * height {
		copy(frame.Pix[at*4:], []byte{255, 255, 255, 255})
	}
	for y := range side.Y {
		for x := range side.X {
			rgb := rendered.Palette[int(rendered.Matrix[y*side.X+x])*3:]
			for py := range gpuPayloadTestModule {
				row := (margin + y*gpuPayloadTestModule + py) * width
				for px := range gpuPayloadTestModule {
					at := (row + margin + x*gpuPayloadTestModule + px) * 4
					copy(frame.Pix[at:], rgb[:3])
					frame.Pix[at+3] = 255
				}
			}
		}
	}
	centre := func(mx, my float64) core.PointF {
		return core.Pt(
			float64(margin)+mx*gpuPayloadTestModule,
			float64(margin)+my*gpuPayloadTestModule,
		)
	}
	return gpuPayloadFixture{
		frame: frame,
		side:  side,
		quad: [4]core.PointF{
			centre(3.5, 3.5),
			centre(float64(side.X)-3.5, 3.5),
			centre(float64(side.X)-3.5, float64(side.Y)-3.5),
			centre(3.5, float64(side.Y)-3.5),
		},
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
// unmasking, bit packing, deinterleaving and hard correction together.
//
// Hard LDPC carries no payload integrity check, so a chain that classifies one
// module differently can still return a plausible payload with no error. The
// comparison is therefore the decoded bytes in full, on a symbol whose payload
// is known, and both arms must reach success rather than agreeing on failure.
func TestGPUPayloadChainMatchesHost(t *testing.T) {
	payload := bytes.Repeat([]byte("device payload chain "), 4)
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

			hostSymbol := &core.DecodedSymbol{}
			hostObs, ret := decode.ObservePrimary(matrix, hostSymbol)
			if ret != core.Success || hostObs == nil {
				t.Fatalf("host observation of the sampled grid failed: %d", ret)
			}
			if got := hostObs.CorrectPayload(); got != core.Success {
				t.Fatalf("host payload correction failed: %d", got)
			}

			deviceSymbol := &core.DecodedSymbol{}
			deviceObs, ret := decode.ObservePrimary(matrix, deviceSymbol)
			if ret != core.Success || deviceObs == nil {
				t.Fatalf("device-arm observation of the sampled grid failed: %d", ret)
			}
			deviceObs.UseDevice(resident)
			// A rebuilt permutation is what proves the device chain ran: nothing
			// else on either arm records one, so a silent decline would leave
			// this at zero and the comparison below would be host against host.
			resident.permutationLength = 0
			if got := deviceObs.CorrectPayload(); got != core.Success {
				t.Fatalf("device payload correction failed: %d", got)
			}
			if resident.permutationLength == 0 {
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
