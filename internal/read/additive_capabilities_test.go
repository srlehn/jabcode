//go:build jabcode_high_color && jabcode_legacy

package read

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestAdditiveCapabilitiesDecodeExistingHighColorSources(t *testing.T) {
	for _, tc := range []struct {
		file   string
		header string
	}{
		{"16c_ecc10_v15_lorem_ms4.png", "JAB high-colour capture test | colors=16"},
		{"256c_ecc10_v9_lorem_ms6.png", "JAB high-colour capture test | colors=256"},
	} {
		t.Run(tc.file, func(t *testing.T) {
			path := filepath.Join(testutil.CapturePath(t), "source", tc.file)
			f, err := os.Open(path)
			if err != nil {
				t.Skipf("open private capture source: %v", err)
			}
			defer f.Close()
			img, err := png.Decode(f)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Decode(img)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(string(got), tc.header) {
				t.Fatalf("Decode prefix = %q, want %q", got[:min(len(got), len(tc.header))], tc.header)
			}
		})
	}
}

func TestStreamSwitchesAcrossCompiledCapabilities(t *testing.T) {
	isoMessage := []byte("stream capability transition ISO")
	isoImage, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, isoMessage)
	if err != nil {
		t.Fatal(err)
	}
	highColorMessage := []byte("stream capability transition high color")
	highColorImage, err := encode.Run(encode.Config{
		Colors: 16, ModuleSize: 12, SymbolNumber: 1, Format: wire.EncodeISOHighColor,
	}, highColorMessage)
	if err != nil {
		t.Fatal(err)
	}

	sequence := []struct {
		name string
		img  image.Image
		want []byte
	}{
		{name: "ISO", img: isoImage, want: isoPayload(isoMessage)},
		{name: "current C", img: loadLegacyCReferenceFixture(t, "c_encoded.png"), want: []byte("Encoded by C, decoded by Go")},
		{name: "BSI", img: loadLegacyCReferenceFixture(t, "bsi_tr_03137_8c_rect_3x2.png"), want: []byte("BSI fixed 3x2 oracle")},
		{name: "pre-v2.0 C", img: loadLegacyCReferenceFixture(t, "legacy_c_reference_pre_v2_8c.png"), want: []byte("Legacy C-reference JAB Code primary fixture 0123456789")},
		{name: "high color", img: highColorImage, want: isoPayload(highColorMessage)},
	}

	var stream Stream
	for _, frame := range sequence {
		t.Run(frame.name, func(t *testing.T) {
			got, frames := requireStreamDecode(t, &stream, frame.img, 5)
			if string(got) != string(frame.want) {
				t.Fatalf("Stream = %q after %d frames, want %q", got, frames, frame.want)
			}
		})
	}
}
