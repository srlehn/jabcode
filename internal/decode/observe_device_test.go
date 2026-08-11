package decode

import (
	"bytes"
	"fmt"
	"image"
	"reflect"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// stubMetadataDevice answers with metadata a caller supplies, so the narrowing
// this package does can be tested without a device.
type stubMetadataDevice struct {
	meta core.PrimaryMetadata
	err  error
}

type stubPrimaryDevice struct {
	result core.PrimaryDeviceResult
	err    error
}

func (stub stubPrimaryDevice) DecodePrimary(
	*core.Bitmap, *core.DecodedSymbol,
) (core.PrimaryDeviceResult, error) {
	return stub.result, stub.err
}

func (stub stubMetadataDevice) WalkPrimaryMetadata(
	*core.Bitmap, *core.DecodedSymbol,
) (core.PrimaryMetadata, error) {
	return stub.meta, stub.err
}

func renderPrimary(t *testing.T, colors int, payload []byte) *core.Bitmap {
	t.Helper()
	r, err := encode.Render(
		encode.Config{Colors: colors, ModuleSize: 1, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("%d colours: render: %v", colors, err)
	}
	bm := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, idx := range r.Matrix {
		copy(bm.Pix[i*4:], r.Palette[int(idx)*3:int(idx)*3+3])
		bm.Pix[i*4+3] = 255
	}
	return bm
}

// narrowMetadata is what a device stage reports: the colour mode, the palette
// bytes and the declared shape, and nothing a host can rederive.
func narrowMetadata(obs *PrimaryObservation) core.PrimaryMetadata {
	symbol := obs.Symbol
	return core.PrimaryMetadata{
		NC:               symbol.Meta.NC,
		Colors:           1 << (symbol.Meta.NC + 1),
		SideVersion:      symbol.Meta.SideVersion,
		ECL:              symbol.Meta.ECL,
		MaskType:         symbol.Meta.MaskType,
		Palette:          append([]byte(nil), symbol.Palette...),
		MetadataModules:  obs.metaModules,
		PartISyndromeOK:  obs.PartISyndromeOK,
		PartIISyndromeOK: obs.PartIISyndromeOK,
	}
}

// TestObservePrimaryOnDeviceMatchesHost holds the device observation to the
// host one on every input payload correction consumes.
//
// The comparison is field by field and ends in the corrected payload rather
// than in a success flag, because hard LDPC has no payload integrity check: a
// reserved map off by one module, or a normalized palette that classifies one
// module differently, produces a plausible wrong answer that a status
// comparison would call a match.
func TestObservePrimaryOnDeviceMatchesHost(t *testing.T) {
	for _, colors := range []int{4, 8} {
		t.Run(fmt.Sprintf("%dc", colors), func(t *testing.T) {
			payload := []byte("device metadata narrows to the same observation")
			bm := renderPrimary(t, colors, payload)

			hostSym := &core.DecodedSymbol{}
			var hostTrace PrimaryTrace
			host, ret := ObservePrimaryTraced(bm, hostSym, &hostTrace)
			if ret != core.Success || host == nil {
				t.Fatalf("host observation failed: %d", ret)
			}

			device := stubMetadataDevice{meta: narrowMetadata(host)}
			deviceSym := &core.DecodedSymbol{}
			var deviceTrace PrimaryTrace
			got, ret, handled := ObservePrimaryOnDevice(device, bm, deviceSym, &deviceTrace)
			if !handled {
				t.Fatal("device observation declined a symbol it described")
			}
			if ret != core.Success || got == nil {
				t.Fatalf("device observation failed: %d", ret)
			}

			if deviceSym.Meta != hostSym.Meta {
				t.Errorf("metadata %+v, want %+v", deviceSym.Meta, hostSym.Meta)
			}
			if deviceSym.SideSize != hostSym.SideSize {
				t.Errorf("side size %v, want %v", deviceSym.SideSize, hostSym.SideSize)
			}
			if !bytes.Equal(deviceSym.Palette, hostSym.Palette) {
				t.Error("palette differs")
			}
			if got.metaModules != host.metaModules {
				t.Errorf("metadata modules %d, want %d", got.metaModules, host.metaModules)
			}
			// Without this the map comparison below would pass on two empty
			// maps, which is the shape a replay bug most easily takes.
			reserved := 0
			for _, mark := range host.dataMap {
				if mark != 0 {
					reserved++
				}
			}
			if reserved != host.metaModules {
				t.Fatalf("host reserved %d modules for a %d-module walk", reserved, host.metaModules)
			}
			if !bytes.Equal(got.dataMap, host.dataMap) {
				differing := 0
				for i := range host.dataMap {
					if got.dataMap[i] != host.dataMap[i] {
						differing++
					}
				}
				t.Errorf("reserved map differs in %d modules", differing)
			}
			if !equalFloats(got.normPalette, host.normPalette) {
				t.Error("normalized palette differs")
			}
			if !equalFloats(got.palThs, host.palThs) {
				t.Error("palette thresholds differ")
			}
			if got.PartISyndromeOK != host.PartISyndromeOK ||
				got.PartIISyndromeOK != host.PartIISyndromeOK {
				t.Error("syndrome verdicts differ")
			}
			// The diagnostics draw one image per stage, so the replayed
			// per-stage maps have to be the maps the stages actually produced
			// and not three copies of the final one.
			for _, stage := range []struct {
				name       string
				host, want []byte
			}{
				{"part I", hostTrace.PartIDataMap, deviceTrace.PartIDataMap},
				{"palette", hostTrace.PaletteDataMap, deviceTrace.PaletteDataMap},
				{"part II", hostTrace.PartIIDataMap, deviceTrace.PartIIDataMap},
			} {
				if !bytes.Equal(stage.host, stage.want) {
					t.Errorf("%s trace map differs", stage.name)
				}
			}

			if res := host.CorrectPayload(); res != core.Success {
				t.Fatalf("host payload correction failed: %d", res)
			}
			if res := got.CorrectPayload(); res != core.Success {
				t.Fatalf("device payload correction failed: %d", res)
			}
			if !bytes.Equal(deviceSym.Data, hostSym.Data) {
				t.Error("corrected payload differs")
			}
			if len(hostSym.Data) == 0 {
				t.Fatal("host correction produced no payload, so the comparison proved nothing")
			}
		})
	}
}

func TestDecodePrimaryOnDeviceResult(t *testing.T) {
	bm := renderPrimary(t, 4, []byte("fused primary result"))
	reference, ret := ObservePrimary(bm, &core.DecodedSymbol{})
	if ret != core.Success {
		t.Fatalf("host observation failed: %d", ret)
	}
	meta := narrowMetadata(reference)
	stream := []byte{1, 0, 1, 1, 0, 0, 0, 0, 1}
	evidence := core.PrimaryEvidence{
		Available: true, MetadataExplicit: true,
		PaletteSeparation: 32,
	}

	symbol := &core.DecodedSymbol{}
	ret, handled := DecodePrimaryOnDevice(stubPrimaryDevice{
		result: core.PrimaryDeviceResult{
			Metadata: meta, Payload: stream, PayloadOK: true, Evidence: evidence,
		},
	}, bm, symbol)
	if !handled || ret != core.Success || !bytes.Equal(symbol.Data, stream[:4]) {
		t.Fatalf("fused result = ret %d handled %v data %v", ret, handled, symbol.Data)
	}

	symbol = &core.DecodedSymbol{}
	ret, handled = DecodePrimaryOnDevice(stubPrimaryDevice{
		result: core.PrimaryDeviceResult{Metadata: meta, PayloadOK: false, Evidence: evidence},
	}, bm, symbol)
	if !handled || ret != core.Failure {
		t.Fatalf("answered payload failure = ret %d handled %v", ret, handled)
	}

	ret, handled = DecodePrimaryOnDevice(stubPrimaryDevice{err: fmt.Errorf("declined")}, bm, &core.DecodedSymbol{})
	if handled || ret != core.Failure {
		t.Fatalf("device decline = ret %d handled %v", ret, handled)
	}
}

// TestObservePrimaryOnDeviceLadders pins what a device answer that is not a
// clean read does to the host path. A decline has to leave the read exactly as
// it would have been, and a rejection has to keep the declared version, which
// is the only thing an alignment-pattern retry has left to resample at.
func TestObservePrimaryOnDeviceLadders(t *testing.T) {
	bm := renderPrimary(t, 8, []byte("ladders"))
	reference := &core.DecodedSymbol{}
	host, ret := ObservePrimary(bm, reference)
	if ret != core.Success {
		t.Fatalf("host observation failed: %d", ret)
	}

	declines := map[string]stubMetadataDevice{
		"error":     {err: fmt.Errorf("declined")},
		"defaulted": {meta: core.PrimaryMetadata{Defaulted: true}},
		"nopalette": {meta: core.PrimaryMetadata{NC: 2, Colors: 8}},
	}
	for name, device := range declines {
		t.Run(name, func(t *testing.T) {
			symbol := &core.DecodedSymbol{}
			obs, ret, handled := ObservePrimaryOnDevice(device, bm, symbol, nil)
			if handled || obs != nil {
				t.Fatalf("device answered a case it cannot: handled=%v", handled)
			}
			if ret != core.Failure {
				t.Errorf("declined with %d, want %d", ret, core.Failure)
			}
			// The host walk runs next over the same symbol, so a decline must
			// not have left anything behind for it to inherit.
			if !reflect.DeepEqual(symbol, &core.DecodedSymbol{}) {
				t.Errorf("a decline wrote to the symbol: %+v", symbol)
			}
		})
	}

	t.Run("rejected", func(t *testing.T) {
		meta := narrowMetadata(host)
		meta.Rejected = true
		meta.SideVersion = image.Pt(7, 7)
		symbol := &core.DecodedSymbol{}
		obs, ret, handled := ObservePrimaryOnDevice(stubMetadataDevice{meta: meta}, bm, symbol, nil)
		if !handled {
			t.Fatal("a rejection is an answer, not a decline")
		}
		if obs != nil || ret != core.Failure {
			t.Fatalf("rejection returned obs=%v ret=%d", obs, ret)
		}
		if symbol.Meta.SideVersion != meta.SideVersion {
			t.Errorf("declared version %v, want %v", symbol.Meta.SideVersion, meta.SideVersion)
		}
		if symbol.SideSize != image.Pt(45, 45) {
			t.Errorf("side size %v, want the declared version's 45x45", symbol.SideSize)
		}
	})
}

func equalFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// stubGridDevice fills a shape-only matrix from a grid it was given, or refuses
// when it was given none.
type stubGridDevice struct {
	pix   []byte
	calls int
}

func (stub *stubGridDevice) MaterializeGrid(matrix *core.Bitmap) bool {
	stub.calls++
	if stub.pix == nil {
		return false
	}
	matrix.Pix = stub.pix
	return true
}

type stubPayloadDevice struct {
	calls          int
	ok             bool
	err            error
	fixedAdmission bool
	request        core.PayloadRequest
}

func (stub *stubPayloadDevice) SupportsFixedPatternAdmission() bool {
	return stub.fixedAdmission
}

func (stub *stubPayloadDevice) CorrectSymbolPayload(request core.PayloadRequest) ([]byte, bool, error) {
	stub.calls++
	stub.request = request
	return nil, stub.ok, stub.err
}

func TestDeviceFixedAdmissionStaysResident(t *testing.T) {
	full := renderPrimary(t, 8, []byte("resident fixed-pattern admission"))
	reference, ret := ObservePrimary(full, &core.DecodedSymbol{})
	if ret != core.Success {
		t.Fatalf("host observation failed: %d", ret)
	}
	meta := narrowMetadata(reference)
	meta.PartIISyndromeOK = false

	observe := func(t *testing.T, pixels []byte, correction *stubPayloadDevice) (*PrimaryObservation, *stubGridDevice) {
		t.Helper()
		matrix := &core.Bitmap{Width: full.Width, Height: full.Height, Channels: full.Channels}
		obs, ret, handled := ObservePrimaryOnDevice(
			stubMetadataDevice{meta: meta}, matrix, &core.DecodedSymbol{}, nil)
		if !handled || ret != core.Success {
			t.Fatalf("device observation failed: ret=%d handled=%v", ret, handled)
		}
		grid := &stubGridDevice{pix: pixels}
		obs.UseGrid(grid)
		obs.UseDevice(correction)
		return obs, grid
	}

	t.Run("answered", func(t *testing.T) {
		correction := &stubPayloadDevice{fixedAdmission: true}
		obs, grid := observe(t, full.Pix, correction)
		if !obs.AdmitPayloadCorrection() {
			t.Fatal("coherent device observation was not admitted")
		}
		if grid.calls != 0 {
			t.Fatalf("admission materialized the grid %d times", grid.calls)
		}
		if result := obs.CorrectPayload(); result != core.Failure {
			t.Fatalf("failed resident correction returned %d", result)
		}
		if !correction.request.RequireFixedPatternAgreement {
			t.Error("resident correction was not asked to gate on fixed patterns")
		}
		if grid.calls != 0 {
			t.Errorf("answered correction materialized the grid %d times", grid.calls)
		}
	})

	t.Run("decline", func(t *testing.T) {
		correction := &stubPayloadDevice{
			fixedAdmission: true,
			err:            fmt.Errorf("declined"),
		}
		obs, grid := observe(t, full.Pix, correction)
		if !obs.AdmitPayloadCorrection() {
			t.Fatal("coherent device observation was not admitted")
		}
		if result := obs.CorrectPayload(); result != core.Success {
			t.Fatalf("host fallback after device decline returned %d", result)
		}
		if grid.calls != 1 {
			t.Errorf("device decline materialized the grid %d times, want 1", grid.calls)
		}
	})

	t.Run("decline rejects fixed patterns", func(t *testing.T) {
		correction := &stubPayloadDevice{
			fixedAdmission: true,
			err:            fmt.Errorf("declined"),
		}
		obs, grid := observe(t, make([]byte, len(full.Pix)), correction)
		if !obs.AdmitPayloadCorrection() {
			t.Fatal("palette half of admission unexpectedly rejected")
		}
		if result := obs.CorrectPayload(); result != core.Failure {
			t.Fatalf("bad fixed patterns returned %d", result)
		}
		if grid.calls != 1 {
			t.Errorf("device decline materialized the grid %d times, want 1", grid.calls)
		}
		if len(obs.Symbol.Data) != 0 {
			t.Error("fixed-pattern rejection reached host payload correction")
		}
	})
}

// TestDevicePayloadFailureDoesNotMaterializeGrid distinguishes an answered
// correction failure from a device decline. Once both resident correction
// stages gave up, replaying the same sample on the host would add the second
// transfer this route exists to remove.
func TestDevicePayloadFailureDoesNotMaterializeGrid(t *testing.T) {
	full := renderPrimary(t, 8, []byte("resident correction failure"))
	reference, ret := ObservePrimary(full, &core.DecodedSymbol{})
	if ret != core.Success {
		t.Fatalf("host observation failed: %d", ret)
	}
	matrix := &core.Bitmap{Width: full.Width, Height: full.Height, Channels: full.Channels}
	obs, ret, handled := ObservePrimaryOnDevice(
		stubMetadataDevice{meta: narrowMetadata(reference)},
		matrix,
		&core.DecodedSymbol{},
		nil,
	)
	if !handled || ret != core.Success {
		t.Fatalf("device observation failed: ret=%d handled=%v", ret, handled)
	}
	grid := &stubGridDevice{pix: full.Pix}
	correction := &stubPayloadDevice{}
	obs.UseGrid(grid)
	obs.UseDevice(correction)
	if result := obs.CorrectPayload(); result != core.Failure {
		t.Fatalf("failed resident correction returned %d", result)
	}
	if correction.calls != 1 {
		t.Errorf("resident correction calls %d, want 1", correction.calls)
	}
	if grid.calls != 0 {
		t.Errorf("answered failure materialized the grid %d times", grid.calls)
	}
}

// TestShapeOnlyGridFailsClosed holds every host stage that reads a sampled
// module to the never-panic contract when the modules are still on a device.
//
// A device-route observation carries only the grid's shape, so a stage that
// indexed Pix without asking would read an empty slice: not a crash and not an
// error, just a symbol made of zeros classified with full confidence. Each
// stage has to refuse instead, and each has to come back once the grid can be
// materialized.
func TestShapeOnlyGridFailsClosed(t *testing.T) {
	full := renderPrimary(t, 8, []byte("shape only"))
	symbol := &core.DecodedSymbol{}
	reference, ret := ObservePrimary(full, symbol)
	if ret != core.Success {
		t.Fatalf("host observation failed: %d", ret)
	}
	modules := append([]byte(nil), full.Pix...)

	shapeOnly := func() *core.Bitmap {
		return &core.Bitmap{Width: full.Width, Height: full.Height, Channels: full.Channels}
	}
	observe := func(grid core.GridDevice) *PrimaryObservation {
		t.Helper()
		matrix := shapeOnly()
		sym := &core.DecodedSymbol{}
		obs, ret, handled := ObservePrimaryOnDevice(
			stubMetadataDevice{meta: narrowMetadata(reference)}, matrix, sym, nil)
		if !handled || ret != core.Success {
			t.Fatalf("device observation on a shape-only grid: ret=%d handled=%v", ret, handled)
		}
		obs.UseGrid(grid)
		return obs
	}

	t.Run("refused", func(t *testing.T) {
		refusing := &stubGridDevice{}
		obs := observe(refusing)
		if _, checked := obs.FixedPatternAgreement(); checked != 0 {
			t.Errorf("classified %d fixed modules with none to read", checked)
		}
		if costs := obs.ModuleCosts(3, 3, nil); len(costs) != 0 {
			t.Errorf("produced %d module costs with none to read", len(costs))
		}
		if snap := obs.Snapshot(); snap != nil {
			t.Error("snapshotted an observation with no modules")
		}
		if res := obs.CorrectPayload(); res == core.Success {
			t.Error("corrected a payload with no modules")
		}
		if res := obs.CorrectPayloadMergedPalette(); res == core.Success {
			t.Error("corrected a merged-palette payload with no modules")
		}
		if refusing.calls == 0 {
			t.Fatal("no stage asked for the modules, so nothing above was tested")
		}
	})

	t.Run("materialized", func(t *testing.T) {
		obs := observe(&stubGridDevice{pix: modules})
		agree, checked := obs.FixedPatternAgreement()
		wantAgree, wantChecked := reference.FixedPatternAgreement()
		if agree != wantAgree || checked != wantChecked {
			t.Errorf("fixed patterns %d/%d, want %d/%d", agree, checked, wantAgree, wantChecked)
		}
		if res := obs.CorrectPayload(); res != core.Success {
			t.Fatalf("payload correction after materializing: %d", res)
		}
		if res := reference.CorrectPayload(); res != core.Success {
			t.Fatalf("reference payload correction: %d", res)
		}
		if !bytes.Equal(obs.Symbol.Data, symbol.Data) {
			t.Error("payload differs from the host observation's")
		}
	})
}
