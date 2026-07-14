package jabcode

import (
	"bytes"
	"errors"
	"image"
	"testing"
)

func TestDefaultProfileIsExperimentalISO23634(t *testing.T) {
	payload := []byte("default ISO profile")
	img, err := NewEncoder().Encode(payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := Decode(img)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	want := isoReaderTransmission(payload)
	if !bytes.Equal(got, want) {
		t.Fatalf("Decode = %q, want %q", got, want)
	}
}

func TestOptionalProfileAvailabilityErrors(t *testing.T) {
	if !ProfileHighColor.Available() {
		_, err := NewEncoder(WithProfile(ProfileHighColor), WithColors(16)).Encode([]byte("high color"))
		if !errors.Is(err, ErrProfileUnavailable) {
			t.Fatalf("high-color encode error = %v, want ErrProfileUnavailable", err)
		}
	}
	if ProfileBSI.Available() {
		t.Fatal("BSI profile became available without its exact implementation")
	}
	if _, err := NewEncoder(WithProfile(ProfileBSI)).Encode([]byte("BSI")); !errors.Is(err, ErrProfileUnavailable) {
		t.Fatalf("BSI encode error = %v, want ErrProfileUnavailable", err)
	}
	_, err := NewEncoder(WithProfile(ProfileLegacy)).Encode([]byte("legacy"))
	if ProfileLegacy.Available() {
		if !errors.Is(err, ErrProfileReadOnly) {
			t.Fatalf("legacy encode error = %v, want ErrProfileReadOnly", err)
		}
	} else if !errors.Is(err, ErrProfileUnavailable) {
		t.Fatalf("legacy encode error = %v, want ErrProfileUnavailable", err)
	}
}

func TestStreamProfileSelection(t *testing.T) {
	if _, err := NewStreamWithProfile(ProfileISO23634); err != nil {
		t.Fatalf("ISO stream: %v", err)
	}
	if !ProfileHighColor.Available() {
		if _, err := NewStreamWithProfile(ProfileHighColor); !errors.Is(err, ErrProfileUnavailable) {
			t.Fatalf("high-color stream error = %v, want ErrProfileUnavailable", err)
		}
	} else {
		payload := []byte("high-color stream profile")
		img, err := NewEncoder(WithProfile(ProfileHighColor), WithColors(16)).Encode(payload)
		if err != nil {
			t.Fatalf("high-color encode: %v", err)
		}
		stream, err := NewStreamWithProfile(ProfileHighColor)
		if err != nil {
			t.Fatalf("high-color stream: %v", err)
		}
		got, err := stream.Decode(img)
		if err != nil {
			t.Fatalf("high-color stream decode: %v", err)
		}
		if want := isoReaderTransmission(payload); !bytes.Equal(got, want) {
			t.Fatalf("high-color stream Decode = %q, want %q", got, want)
		}
	}
	if ProfileLegacy.Available() {
		if _, err := NewStreamWithProfile(ProfileLegacy); err == nil {
			t.Fatal("legacy stream constructor accepted a profile whose pre-v2.0 finder fallback is unavailable in the bounded scheduler")
		}
	}
}

func TestISOProfileReaderTransmission(t *testing.T) {
	payload := []byte("ISO/IEC 23634 conformance profile 0123456789")
	want := append([]byte("]j1"), payload...)
	for _, colors := range []int{4, 8} {
		img, err := NewEncoder(
			WithColors(colors),
			WithProfile(ProfileISO23634),
		).Encode(payload)
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		got, err := DecodeWithProfile(img, ProfileISO23634)
		if err != nil {
			t.Fatalf("colors %d decode: %v", colors, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("colors %d decoded %q, want %q", colors, got, want)
		}
	}
}

func TestISOProfileMultiSymbolReaderTransmission(t *testing.T) {
	payload := bytes.Repeat([]byte("ISO cascade "), 10)
	want := append([]byte("]j1"), payload...)
	for _, colors := range []int{4, 8} {
		img, err := NewEncoder(
			WithColors(colors),
			WithSymbols(
				[]int{0, 2},
				[]image.Point{image.Pt(4, 4), image.Pt(4, 4)},
				[]int{0, 0},
			),
			WithProfile(ProfileISO23634),
		).Encode(payload)
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		got, err := DecodeWithProfile(img, ProfileISO23634)
		if err != nil {
			t.Fatalf("colors %d decode: %v", colors, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("colors %d decoded %q, want %q", colors, got, want)
		}
	}
}

func TestISOProfileRejectsReservedColorModes(t *testing.T) {
	for _, colors := range []int{16, 32, 64, 128, 256} {
		_, err := NewEncoder(
			WithColors(colors),
			WithProfile(ProfileISO23634),
		).Encode([]byte("reserved"))
		if err == nil {
			t.Errorf("colors %d: expected ISO-profile reserved-mode error", colors)
		}
	}
}

func TestInvalidProfile(t *testing.T) {
	profile := Profile(255)
	if _, err := NewEncoder(WithProfile(profile)).Encode([]byte("invalid")); err == nil {
		t.Error("encoder accepted invalid profile")
	}
	if _, err := DecodeWithProfile(nil, profile); err == nil {
		t.Error("decoder accepted invalid profile")
	}
}
