package encoder_test

import (
	"bytes"
	"image"
	"image/png"
	"testing"

	"github.com/srlehn/jabcode/encoder"
)

func TestPublicEncoderDeterministicFixedVersion(t *testing.T) {
	options := []encoder.Option{
		encoder.WithModuleSize(2),
		encoder.WithSymbols([]int{0}, []image.Point{image.Pt(4, 4)}, []int{0}),
	}
	payload := []byte("dependency-light deterministic encoder")

	first, err := encoder.New(options...).Encode(payload)
	if err != nil {
		t.Fatalf("first encode: %v", err)
	}
	second, err := encoder.New(options...).Encode(payload)
	if err != nil {
		t.Fatalf("second encode: %v", err)
	}
	if got, want := first.Bounds().Size(), image.Pt(66, 66); got != want {
		t.Fatalf("fixed-version geometry = %v, want %v", got, want)
	}

	var firstPNG, secondPNG bytes.Buffer
	if err := png.Encode(&firstPNG, first); err != nil {
		t.Fatalf("encode first PNG: %v", err)
	}
	if err := png.Encode(&secondPNG, second); err != nil {
		t.Fatalf("encode second PNG: %v", err)
	}
	if !bytes.Equal(firstPNG.Bytes(), secondPNG.Bytes()) {
		t.Fatal("identical public encoder calls produced different images")
	}
}

func TestPublicEncoderRejectsInvalidFixedVersion(t *testing.T) {
	_, err := encoder.New(
		encoder.WithSymbols([]int{0}, []image.Point{image.Pt(33, 4)}, []int{0}),
	).Encode([]byte("invalid version"))
	if err == nil {
		t.Fatal("version 33 was accepted")
	}
}
