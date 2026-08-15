//go:build !js

package detect

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"math/bits"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
)

// mapPayloadShape records the resident data map for one symbol shape and
// returns the payload and correction control it derived from it. It is a
// parity seam, not production readback: production keeps both blocks on the
// device and the host never learns the shape.
func (resident *gpuResidentBinarizer) mapPayloadShape(params []byte) ([]byte, []byte, error) {
	resident.mu.Lock()
	defer resident.mu.Unlock()

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, nil, err
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.payloadParams, 0, params); err != nil {
		return nil, nil, err
	}
	if err := recorder.Fill(resident.ldpcParams, 0, gpuLDPCParamWords*4, 0); err != nil {
		return nil, nil, err
	}
	if err := recorder.Barrier(resident.payloadParams, resident.ldpcParams); err != nil {
		return nil, nil, err
	}
	if err := recordGPUOneWorkgroup(
		recorder, resident.payloadMapKernel, resident.payloadMapBindings, nil,
	); err != nil {
		return nil, nil, err
	}
	if err := recorder.Barrier(resident.payloadParams, resident.ldpcParams); err != nil {
		return nil, nil, err
	}
	payload := make([]byte, gpuPayloadParamWords*4)
	ldpc := make([]byte, gpuLDPCParamWords*4)
	if err := recorder.Download(resident.payloadParams, 0, payload); err != nil {
		return nil, nil, err
	}
	if err := recorder.Download(resident.ldpcParams, 0, ldpc); err != nil {
		return nil, nil, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, nil, err
	}
	return payload, ldpc, nil
}

// TestGPUPayloadShapeDerivationMatchesHost holds the device's own key
// derivation to the host split.
//
// The catalog lookup is gated separately, over the whole key space, but the key
// it looks up is derived here: the data map counts the symbol's payload modules
// and turns them, with the colour depth and the ECC pair, into a codeword length
// and a trailing length. A shape the device derives differently selects another
// code, and hard LDPC has no payload integrity check to report that downstream.
// The data-module count is compared against the host observer's own reserved
// map rather than against the number handed in, so both halves of the
// derivation are covered.
func TestGPUPayloadShapeDerivationMatchesHost(t *testing.T) {
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
			t.Errorf("close GPU payload shape device: %v", err)
		}
	})

	payload := bytes.Repeat([]byte("payload shape "), 8)
	// A square symbol at one size cannot show a version pair being read from the
	// wrong axis, and a small one never reaches a trailing block, so the sweep
	// crosses the colour depths and ECC levels with a few explicit shapes.
	shapes := []image.Point{{}, {X: 6, Y: 14}, {X: 14, Y: 6}, {X: 21, Y: 21}}
	for _, colors := range []int{4, 8, 16, 32, 64, 128, 256} {
		for _, level := range []int{6, 10, spec.DefaultECCLevel} {
			for _, shape := range shapes {
				runGPUPayloadShapeCase(t, resident, colors, level, shape, payload)
			}
		}
	}
}

func runGPUPayloadShapeCase(
	t *testing.T,
	resident *gpuResidentBinarizer,
	colors, level int,
	shape image.Point,
	payload []byte,
) {
	name := fmt.Sprintf("colors=%d/ecc=%d/version=%dx%d", colors, level, shape.X, shape.Y)
	t.Run(name, func(t *testing.T) {
		fixture := gpuPayloadRenderVersion(t, colors, level, shape, payload)
		symbol := &core.DecodedSymbol{WireVariant: gpuPayloadVariant(colors)}
		obs, result := decode.ObservePrimary(fixture.modules, symbol)
		if result != core.Success || obs == nil {
			t.Fatalf("host observation declined the fixture: %d", result)
		}
		snapshot := obs.Snapshot()
		if snapshot == nil {
			t.Fatal("host observation produced no snapshot")
		}
		dataModules := 0
		for _, reserved := range snapshot.DataMap {
			if reserved == 0 {
				dataModules++
			}
		}
		copies := spec.PaletteCopies(colors)
		shape, err := gpuPayloadShapeOf(core.PayloadRequest{
			Matrix:            fixture.modules,
			Symbol:            symbol,
			MetadataModules:   obs.MetadataModules(),
			DataModules:       dataModules,
			NormalizedPalette: make([]float64, colors*4*copies),
			PaletteThresholds: make([]float64, 3*spec.ColorPaletteNumber),
		})
		if err != nil {
			// A trailing block past MaxCapacity is refused by both routes before
			// any key is looked up, which large high-colour symbols reach. There
			// is no device shape to compare against for those.
			t.Skipf("host payload shape declined this case: %v", err)
		}
		gotPayload, gotLDPC, err := resident.mapPayloadShape(shape.params[:])
		if err != nil {
			t.Fatalf("device payload shape: %v", err)
		}
		word := func(raw []byte, index int) int {
			return int(binary.LittleEndian.Uint32(raw[index*4:]))
		}
		bitsPerModule := bits.Len(uint(colors)) - 1
		wc, wr := symbol.Meta.ECL.X, symbol.Meta.ECL.Y
		layout := ecc.HardBlockSplit(dataModules*bitsPerModule, wc, wr)
		for _, check := range []struct {
			what string
			got  int
			want int
		}{
			{"data modules", word(gotPayload, gpuPayloadParamDataModules), dataModules},
			{"gross bits", word(gotPayload, gpuPayloadParamGrossBits), layout.Pg},
			{"net bits", word(gotPayload, gpuPayloadParamNetBits), layout.Pn},
			{"length", word(gotLDPC, gpuLDPCParamLength), layout.GrossSub},
			{"net length", word(gotLDPC, gpuLDPCParamNet), layout.NetSub},
			{"blocks", word(gotLDPC, gpuLDPCParamBlocks), layout.Blocks},
			{"trailing length", word(gotLDPC, gpuLDPCParamTailLength), trailingGross(layout)},
		} {
			if check.got != check.want {
				t.Fatalf("device %s = %d, host %d (side %v, wc %d, wr %d)",
					check.what, check.got, check.want, symbol.SideSize, wc, wr)
			}
		}
	})
}

// gpuPayloadRenderVersion is gpuPayloadRender at an explicit version pair, or
// the encoder's own choice when the pair is zero.
func gpuPayloadRenderVersion(
	t *testing.T,
	colors, eccLevel int,
	version image.Point,
	payload []byte,
) gpuPayloadFixture {
	t.Helper()
	if version.X == 0 || version.Y == 0 {
		return gpuPayloadRender(t, colors, eccLevel, payload)
	}
	rendered, err := encode.Render(encode.Config{
		Colors: colors, ModuleSize: 1, ECCLevel: eccLevel, SymbolNumber: 1,
		SymbolVersions: []image.Point{version},
	}, payload)
	if err != nil {
		t.Skipf("render %d-colour %v symbol: %v", colors, version, err)
	}
	return gpuPayloadFixtureOf(t, rendered, colors, eccLevel)
}

// trailingGross is the split's trailing block length, or zero when every block
// is the same size.
func trailingGross(layout ecc.HardBlockLayout) int {
	if layout.Uniform {
		return 0
	}
	return layout.TrailingGrossSub()
}
