//go:build jabcode_legacy

package read

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

func TestCurrentCFallbackReusesNeutralModuleEvidence(t *testing.T) {
	img := loadLegacyCReferenceFixture(t, "c_encoded.png")
	want := []byte("Encoded by C, decoded by Go")
	capabilities := wire.ISO23634.Mask() | wire.CurrentC.Mask()
	got, trace, err := DecodeWithTraceCapabilities(img, capabilities)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("DecodeWithTraceCapabilities = %q, %v; want %q", got, err, want)
	}

	var decodedAttempt *DiagnosticAttempt
	for i := range trace.Attempts {
		if len(trace.Attempts[i].Primary) >= 2 && bytes.Equal(trace.Attempts[i].Payload, want) {
			decodedAttempt = &trace.Attempts[i]
			break
		}
	}
	if decodedAttempt == nil {
		t.Fatal("decoded trace has no ISO/current-C observation pair")
	}
	primary := decodedAttempt.Primary
	if len(primary) < 2 {
		t.Fatal("decoded trace has no ISO/current-C observation pair")
	}
	if primary[0].Classification.ReusedEvidence {
		t.Fatal("first current-family interpretation reported reused evidence")
	}
	if !primary[1].Classification.ReusedEvidence {
		t.Fatal("current-C fallback reclassified the shared module observation")
	}
	if !bytes.Equal(primary[0].Classification.DataMap, primary[1].Classification.DataMap) ||
		!bytes.Equal(primary[0].Classification.Colors, primary[1].Classification.Colors) {
		t.Fatal("reused current-C module evidence differs from the ISO observation")
	}
	if len(decodedAttempt.Alignments) != 1 {
		t.Fatalf("alignment samples = %d, want one", len(decodedAttempt.Alignments))
	}
}
