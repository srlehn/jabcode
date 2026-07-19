package encoder_test

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/encoder"
)

func TestOpaquePlanExactCapacityAndGeometry(t *testing.T) {
	plan, err := encoder.NewOpaquePlan(
		image.Pt(4, 4),
		encoder.WithModuleSize(2),
	)
	if err != nil {
		t.Fatalf("NewOpaquePlan: %v", err)
	}
	if got, want := plan.Colors(), 8; got != want {
		t.Fatalf("Colors = %d, want %d", got, want)
	}
	if got, want := plan.Version(), image.Pt(4, 4); got != want {
		t.Fatalf("Version = %v, want %v", got, want)
	}
	if got, want := plan.ECCLevel(), 3; got != want {
		t.Fatalf("ECCLevel = %d, want %d", got, want)
	}
	if got, want := plan.ModuleSize(), 2; got != want {
		t.Fatalf("ModuleSize = %d, want %d", got, want)
	}
	if got, want := plan.ModuleDimensions(), image.Pt(33, 33); got != want {
		t.Fatalf("ModuleDimensions = %v, want %v", got, want)
	}
	if got, want := plan.ImageDimensions(), image.Pt(66, 66); got != want {
		t.Fatalf("ImageDimensions = %v, want %v", got, want)
	}
	if plan.Capacity() < 16 {
		t.Fatalf("Capacity = %d, want at least the byte-mode length boundary", plan.Capacity())
	}

	for length := 1; length <= plan.Capacity(); length++ {
		payload := make([]byte, length)
		for i := range payload {
			payload[i] = byte(i*131 + 17)
		}
		img, err := plan.Encode(payload)
		if err != nil {
			t.Fatalf("Encode length %d of %d: %v", length, plan.Capacity(), err)
		}
		if got := img.Bounds().Size(); got != plan.ImageDimensions() {
			t.Fatalf("Encode length %d geometry = %v, want %v", length, got, plan.ImageDimensions())
		}
	}

	if _, err := plan.Encode(nil); err == nil {
		t.Fatal("empty opaque payload was accepted")
	}
	if _, err := plan.Encode(make([]byte, plan.Capacity()+1)); err == nil {
		t.Fatal("capacity plus one was accepted")
	}
}

func TestOpaquePlanPinsNonDefaultECC(t *testing.T) {
	plan, err := encoder.NewOpaquePlan(
		image.Pt(6, 5),
		encoder.WithColors(4),
		encoder.WithECCLevel(9),
	)
	if err != nil {
		t.Fatalf("NewOpaquePlan: %v", err)
	}
	if got, want := plan.ECCLevel(), 9; got != want {
		t.Fatalf("ECCLevel = %d, want %d", got, want)
	}
	if _, err := plan.Encode(make([]byte, plan.Capacity())); err != nil {
		t.Fatalf("Encode exact capacity: %v", err)
	}
}

func TestOpaquePlanRejectsDivergentSymbolOptions(t *testing.T) {
	_, err := encoder.NewOpaquePlan(
		image.Pt(4, 4),
		encoder.WithSymbols([]int{0}, []image.Point{image.Pt(3, 3)}, []int{3}),
	)
	if err == nil {
		t.Fatal("WithSymbols was accepted by NewOpaquePlan")
	}
}
