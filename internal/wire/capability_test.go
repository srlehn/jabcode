package wire

import "testing"

func TestCapabilityMask(t *testing.T) {
	capabilities := ISO23634.Mask() | BSI.Mask() | PreV2C.Mask()
	if !capabilities.Valid() {
		t.Fatalf("capability mask %#x is invalid", capabilities)
	}
	if !capabilities.Has(ISO23634) || capabilities.Has(ISOHighColor) ||
		!capabilities.Has(BSI) || !capabilities.Has(PreV2C) || capabilities.Has(CurrentC) {
		t.Fatalf("capability mask membership is wrong: %#x", capabilities)
	}
	if Capabilities(0).Valid() || Capabilities(1<<7).Valid() {
		t.Fatal("invalid decoder capability mask accepted")
	}
}

func TestEncodingVariant(t *testing.T) {
	for encoding, want := range map[Encoding]Variant{
		EncodeISO23634:     ISO23634,
		EncodeISOHighColor: ISOHighColor,
		EncodeCurrentC:     CurrentC,
		EncodeBSI:          BSI,
	} {
		if got := encoding.Variant(); got != want {
			t.Errorf("Encoding(%d).Variant() = %d, want %d", encoding, got, want)
		}
	}
	if Encoding(255).Valid() || Encoding(255).Variant().Valid() {
		t.Fatal("invalid encoding accepted")
	}
}
