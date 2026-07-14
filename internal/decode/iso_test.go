package decode

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestISODefaultSymbolCorrection(t *testing.T) {
	r, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 1, Format: wire.EncodeISO23634, SymbolNumber: 1,
	}, []byte("ISO/IEC 23634 conformance variant 0123456789"))
	if err != nil {
		t.Fatal(err)
	}
	bm := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, c := range r.Matrix {
		copy(bm.Pix[i*4:i*4+3], r.Palette[int(c)*3:int(c)*3+3])
		bm.Pix[i*4+3] = 255
	}
	symbol := &core.DecodedSymbol{WireVariant: wire.ISO23634}
	obs, ret := ObservePrimary(bm, symbol)
	if ret != core.Success {
		t.Fatalf("ObservePrimary returned %d", ret)
	}
	fillDataMap(obs.dataMap, bm.Width, bm.Height, 0)
	rawModules := readRawModuleData(bm, symbol, obs.dataMap, obs.normPalette, obs.palThs)
	demaskSymbol(rawModules, obs.dataMap, symbol.SideSize, symbol.Meta.MaskType, 1<<(symbol.Meta.NC+1))
	raw := rawModuleData2RawData(rawModules, symbol.Meta.NC+1)
	wc, wr := symbol.Meta.ECL.X, symbol.Meta.ECL.Y
	pg := (len(raw) / wr) * wr
	raw = raw[:pg]
	ecc.DeinterleaveVariant(raw, wire.ISO23634)
	if _, ok := ecc.DecodeLDPCHardVariant(raw, wc, wr, wire.ISO23634); !ok {
		t.Fatalf("clean ISO codeword failed syndrome: raw modules=%d gross bits=%d weights=(%d,%d)", len(rawModules), len(raw), wc, wr)
	}
	if got := obs.CorrectPayload(); got != core.Success {
		t.Fatalf("CorrectPayload returned %d", got)
	}
}
