//go:build jabcode_high_color

package read

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestStreamDoesNotAccumulateHighColorBeyondEvidenceScope(t *testing.T) {
	payload := []byte("sixteen-colour stream stays single-frame")
	img, err := encode.Run(encode.Config{
		Colors: 16, ModuleSize: 12, ECCLevel: 3,
		Format: wire.EncodeISOHighColor, SymbolNumber: 1,
	}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var s Stream
	want := isoPayload(payload)
	for i := range 2 {
		got, err := s.Decode(img)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("frame %d = %q, %v", i, got, err)
		}
		if len(s.group.snaps) != 0 {
			t.Fatalf("frame %d retained unsupported colour evidence", i)
		}
	}
}
