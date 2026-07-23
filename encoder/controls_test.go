package encoder_test

import (
	"fmt"
	"image"
	"testing"

	jabcode "github.com/srlehn/jabcode"
	"github.com/srlehn/jabcode/encoder"
)

func TestStructuredECIRoundTrip(t *testing.T) {
	for _, assignment := range []int{0, 127, 128, 16383, 16384, 999999} {
		t.Run(fmt.Sprint(assignment), func(t *testing.T) {
			data := []byte("structured ECI")
			img, err := encoder.New(
				encoder.WithModuleSize(8),
				encoder.WithControls([]encoder.Control{{
					Kind: encoder.ControlECI, Assignment: assignment,
				}}),
			).Encode(data)
			if err != nil {
				t.Fatal(err)
			}
			message, err := jabcode.DecodeMessage(img)
			if err != nil {
				t.Fatal(err)
			}
			if string(message.Data) != string(data) {
				t.Fatalf("data = %q, want %q", message.Data, data)
			}
			if len(message.Controls) != 1 || message.Controls[0].Kind != jabcode.MessageControlECI ||
				message.Controls[0].Offset != 0 || message.Controls[0].Assignment != assignment {
				t.Fatalf("controls = %+v, want ECI assignment %d at offset 0", message.Controls, assignment)
			}
		})
	}
}

func TestStructuredFNC1RoundTrip(t *testing.T) {
	data := []byte("ABC")
	img, err := encoder.New(
		encoder.WithModuleSize(8),
		encoder.WithControls([]encoder.Control{
			{Kind: encoder.ControlFNC1Start, Offset: 0},
			{Kind: encoder.ControlFNC1Separator, Offset: 1},
			{Kind: encoder.ControlFNC1End, Offset: len(data)},
		}),
	).Encode(data)
	if err != nil {
		t.Fatal(err)
	}
	message, err := jabcode.DecodeMessage(img)
	if err != nil {
		t.Fatal(err)
	}
	wantData := []byte{'A', 29, 'B', 'C'}
	if string(message.Data) != string(wantData) {
		t.Fatalf("data = %q, want %q", message.Data, wantData)
	}
	wantKinds := []jabcode.MessageControlKind{
		jabcode.MessageControlFNC1Start,
		jabcode.MessageControlFNC1Separator,
		jabcode.MessageControlFNC1End,
	}
	if len(message.Controls) != len(wantKinds) {
		t.Fatalf("controls = %+v, want %d controls", message.Controls, len(wantKinds))
	}
	for i, want := range wantKinds {
		if message.Controls[i].Kind != want {
			t.Fatalf("control %d = %+v, want kind %d", i, message.Controls[i], want)
		}
	}
}

func TestStructuredControlsRejectInvalidConfiguration(t *testing.T) {
	_, err := encoder.New(
		encoder.WithModuleSize(2),
		encoder.WithControls([]encoder.Control{{Kind: encoder.ControlFNC1End}}),
	).Encode([]byte("data"))
	if err == nil {
		t.Fatal("unterminated FNC1 end was accepted")
	}

	_, err = encoder.New(
		encoder.WithModuleSize(2),
		encoder.WithControls([]encoder.Control{{Kind: encoder.ControlECI, Assignment: 1000000}}),
	).Encode([]byte("data"))
	if err == nil {
		t.Fatal("out-of-range ECI assignment was accepted")
	}

	_, err = encoder.New(
		encoder.WithModuleSize(2),
		encoder.WithSymbols([]int{0, 1}, []image.Point{{5, 5}, {5, 5}}, []int{3, 3}),
		encoder.WithControls([]encoder.Control{{Kind: encoder.ControlECI, Assignment: 26}}),
	).Encode([]byte("data"))
	if err == nil {
		t.Fatal("structured controls were accepted for multi-symbol output")
	}
}
