package jabcode

import (
	"bytes"
	"image"
	"testing"
)

func opaqueByteCorpus() []byte {
	payload := make([]byte, 0, 320)
	for value := range 256 {
		payload = append(payload, byte(value))
	}
	payload = append(payload,
		0, '\\', '\\', 0,
		']', 'j', '1', ']', 'j', '4', ']', 'j', '5',
		'\\', '0', '0', '0', '0', '2', '6',
		29, 30, 4,
	)
	return payload
}

func TestOpaquePlanRawDecodeRoundTrip(t *testing.T) {
	plan, err := NewOpaquePlan(image.Pt(8, 8), WithModuleSize(4))
	if err != nil {
		t.Fatalf("NewOpaquePlan: %v", err)
	}
	payload := opaqueByteCorpus()
	if len(payload) > plan.Capacity() {
		t.Fatalf("test corpus is %d bytes, plan capacity is %d", len(payload), plan.Capacity())
	}
	img, err := plan.Encode(payload)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	message, err := DecodeMessage(img)
	if err != nil {
		t.Fatalf("DecodeMessage: %v", err)
	}
	if !bytes.Equal(message.Data, payload) {
		t.Fatalf("raw data differs: got %d bytes, want %d", len(message.Data), len(payload))
	}
	if len(message.Controls) != 0 {
		t.Fatalf("opaque byte data produced controls: %+v", message.Controls)
	}
	wantTransmission := []byte("]j1")
	for _, value := range payload {
		if value == '\\' {
			wantTransmission = append(wantTransmission, '\\')
		}
		wantTransmission = append(wantTransmission, value)
	}
	if !bytes.Equal(message.ReaderTransmission, wantTransmission) {
		t.Fatal("reader transmission does not preserve the independent ISO representation")
	}
	gotTransmission, err := Decode(img)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if !bytes.Equal(gotTransmission, wantTransmission) {
		t.Fatal("Decode no longer returns the standards-facing reader transmission")
	}
}

func TestOpaquePlanStreamRawDecodeRoundTrip(t *testing.T) {
	plan, err := NewOpaquePlan(image.Pt(8, 8), WithModuleSize(4))
	if err != nil {
		t.Fatalf("NewOpaquePlan: %v", err)
	}
	payload := opaqueByteCorpus()
	img, err := plan.Encode(payload)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	stream := NewStream()
	var message Message
	for frame := 0; frame < 4; frame++ {
		message, err = stream.DecodeMessage(img)
		if err == nil {
			break
		}
	}
	if err != nil {
		t.Fatalf("Stream.DecodeMessage: %v", err)
	}
	if !bytes.Equal(message.Data, payload) {
		t.Fatalf("stream raw data differs: got %d bytes, want %d", len(message.Data), len(payload))
	}
}
