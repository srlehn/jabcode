package read

import (
	"bytes"
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
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

func TestDecodeWithTraceRecordsActualOrientationProbe(t *testing.T) {
	payload := []byte("trace every probe angle")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, trace, err := DecodeWithTrace(detect.RotateImage(img, 30))
	want := isoPayload(payload)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("DecodeWithTrace = (%q,%v), want %q", got, err, want)
	}
	if len(trace.Probes) == 0 {
		t.Fatal("rotated decode recorded no orientation probe")
	}
	for i, probe := range trace.Probes {
		if len(probe.Probe.Angles) != 6 {
			t.Fatalf("probe %d angles = %d, want 6", i, len(probe.Probe.Angles))
		}
		for j, angle := range probe.Probe.Angles {
			if angle.Bitmap == nil || angle.Channels[0] == nil || angle.Channels[1] == nil || angle.Channels[2] == nil {
				t.Fatalf("probe %d angle %d lacks image state", i, j)
			}
		}
	}
	foundWinner := false
	for _, a := range trace.Attempts {
		if a.Stage == readDecoded.String() {
			foundWinner = true
			break
		}
	}
	if !foundWinner {
		t.Fatal("trace omitted the successful route")
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
