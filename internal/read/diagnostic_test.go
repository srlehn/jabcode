package read

import (
	"bytes"
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
)

func TestDecodeWithTraceMatchesDecode(t *testing.T) {
	payload := []byte("single authoritative diagnostic trace")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	want, wantErr := Decode(img)
	got, trace, gotErr := DecodeWithTrace(img)
	if (gotErr != nil) != (wantErr != nil) || !bytes.Equal(got, want) {
		t.Fatalf("DecodeWithTrace = (%q,%v), Decode = (%q,%v)", got, gotErr, want, wantErr)
	}
	if trace == nil {
		t.Fatal("trace is nil")
	}
	if len(trace.Attempts) != 1 {
		t.Fatalf("trace attempts = %d, want 1", len(trace.Attempts))
	}
	a := trace.Attempts[0]
	if a.Stage != readDecoded.String() || a.Sampled == nil || len(a.Primary) != 1 {
		t.Fatalf("upright trace = stage %q sampled=%v primary=%d", a.Stage, a.Sampled != nil, len(a.Primary))
	}
	if !a.Primary[0].CorrectionAttempted || a.Primary[0].CorrectionResult <= 0 {
		t.Fatalf("payload correction trace = attempted %v result %d",
			a.Primary[0].CorrectionAttempted, a.Primary[0].CorrectionResult)
	}
	wantPayload := isoPayload(payload)
	if !bytes.Equal(a.Payload, wantPayload) {
		t.Fatalf("attempt payload = %q, want %q", a.Payload, wantPayload)
	}
}

func TestDecodeWithTraceRecordsDrawableEarlyExit(t *testing.T) {
	_, trace, err := DecodeWithTrace(image.NewNRGBA(image.Rect(0, 0, 64, 64)))
	if err == nil {
		t.Fatal("blank image decoded")
	}
	if trace == nil {
		t.Fatal("trace is nil")
	}
	if len(trace.Attempts) != 1 {
		t.Fatalf("trace attempts = %d, want 1", len(trace.Attempts))
	}
	a := trace.Attempts[0]
	if a.Balanced == nil || a.InitialChannels[0] == nil || len(a.Detector.Passes) == 0 {
		t.Fatalf("early-exit trace lacks drawable state: balanced=%v channels=%v passes=%d",
			a.Balanced != nil, a.InitialChannels[0] != nil, len(a.Detector.Passes))
	}
}
