package wire

import "testing"

func TestProfileMask(t *testing.T) {
	profiles := ISO23634.Mask() | BSI.Mask() | Legacy.Mask()
	if !profiles.Valid() {
		t.Fatalf("profile mask %#x is invalid", profiles)
	}
	if !profiles.Has(ISO23634) || profiles.Has(HighColor) || !profiles.Has(BSI) || !profiles.Has(Legacy) {
		t.Fatalf("profile mask membership is wrong: %#x", profiles)
	}
	if Profiles(0).Valid() || Profiles(1<<7).Valid() {
		t.Fatal("invalid decoder profile mask accepted")
	}
}
