//go:build jabcode_high_color

package read

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestISOAndHighColorShareOneBaseObservation(t *testing.T) {
	capabilities := wire.ISO23634.Mask() | wire.ISOHighColor.Mask()
	for _, tc := range []struct {
		name    string
		colors  int
		format  wire.Encoding
		variant wire.Variant
	}{
		{name: "ISO eight-color", colors: 8, format: wire.EncodeISO23634, variant: wire.ISO23634},
		{name: "ISO high-color", colors: 16, format: wire.EncodeISOHighColor, variant: wire.ISOHighColor},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := []byte("one ISO-base observation")
			want := []byte("]j1one ISO-base observation")
			img, err := encode.Run(encode.Config{Colors: tc.colors, ModuleSize: 12, Format: tc.format, SymbolNumber: 1}, payload)
			if err != nil {
				t.Fatal(err)
			}
			got, trace, err := DecodeWithTraceCapabilities(img, capabilities)
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("DecodeWithTraceCapabilities = %q, %v; want %q", got, err, want)
			}
			for i := range trace.Attempts {
				attempt := &trace.Attempts[i]
				if !bytes.Equal(attempt.Payload, want) {
					continue
				}
				if len(attempt.Primary) != 1 {
					t.Fatalf("primary observations = %d, want one", len(attempt.Primary))
				}
				if gotVariant := attempt.Primary[0].Symbol.WireVariant; gotVariant != tc.variant {
					t.Fatalf("wire variant = %d, want %d", gotVariant, tc.variant)
				}
				return
			}
			t.Fatal("decoded attempt missing from trace")
		})
	}
}

func TestNormalizeCurrentVariantKeepsHighColorFallback(t *testing.T) {
	symbol := core.DecodedSymbol{WireVariant: wire.ISOHighColor}
	symbol.Meta.NC = 3
	lowColor := core.DecodedSymbol{WireVariant: wire.ISOHighColor}
	lowColor.Meta.NC = 2
	highColor := core.DecodedSymbol{WireVariant: wire.ISOHighColor}
	highColor.Meta.NC = 3
	detail := DiagnosticAttempt{Primary: []decode.PrimaryTrace{
		{Symbol: lowColor},
		{Symbol: highColor},
	}}

	normalizeCurrentVariant(&symbol, &detail, wire.ISO23634.Mask()|wire.ISOHighColor.Mask(), 0)
	if symbol.WireVariant != wire.ISOHighColor {
		t.Fatalf("resolved high-color variant = %d, want high-color", symbol.WireVariant)
	}
	if got := detail.Primary[0].Symbol.WireVariant; got != wire.ISO23634 {
		t.Fatalf("low-color observation variant = %d, want ISO", got)
	}
	if got := detail.Primary[1].Symbol.WireVariant; got != wire.ISOHighColor {
		t.Fatalf("high-color fallback variant = %d, want high-color", got)
	}
}
