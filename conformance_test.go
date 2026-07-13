package jabcode

import (
	"bytes"
	"image"
	"testing"
)

func TestISOConformanceRoundTrip(t *testing.T) {
	want := []byte("ISO/IEC 23634 conformance profile 0123456789")
	for _, colors := range []int{4, 8} {
		img, err := NewEncoder(
			WithColors(colors),
			WithConformance(ConformanceISO23634),
		).Encode(want)
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		got, err := DecodeWithConformance(img, ConformanceISO23634)
		if err != nil {
			t.Fatalf("colors %d decode: %v", colors, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("colors %d decoded %q, want %q", colors, got, want)
		}
	}
}

func TestISOConformanceMultiSymbolRoundTrip(t *testing.T) {
	want := bytes.Repeat([]byte("ISO cascade "), 10)
	for _, colors := range []int{4, 8} {
		img, err := NewEncoder(
			WithColors(colors),
			WithSymbols(
				[]int{0, 2},
				[]image.Point{image.Pt(4, 4), image.Pt(4, 4)},
				[]int{0, 0},
			),
			WithConformance(ConformanceISO23634),
		).Encode(want)
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		got, err := DecodeWithConformance(img, ConformanceISO23634)
		if err != nil {
			t.Fatalf("colors %d decode: %v", colors, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("colors %d decoded %q, want %q", colors, got, want)
		}
	}
}

func TestISOConformanceRejectsReservedColorModes(t *testing.T) {
	for _, colors := range []int{16, 32, 64, 128, 256} {
		_, err := NewEncoder(
			WithColors(colors),
			WithConformance(ConformanceISO23634),
		).Encode([]byte("reserved"))
		if err == nil {
			t.Errorf("colors %d: expected strict conformance error", colors)
		}
	}
}

func TestInvalidConformanceMode(t *testing.T) {
	mode := ConformanceMode(255)
	if _, err := NewEncoder(WithConformance(mode)).Encode([]byte("invalid")); err == nil {
		t.Error("encoder accepted invalid conformance mode")
	}
	if _, err := DecodeWithConformance(nil, mode); err == nil {
		t.Error("decoder accepted invalid conformance mode")
	}
}
