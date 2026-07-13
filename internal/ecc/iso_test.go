package ecc

import (
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

func firstNISO(seed uint64, n int) []uint32 {
	out := make([]uint32, 0, n)
	for x := range isoValues(seed) {
		out = append(out, x)
		if len(out) == n {
			break
		}
	}
	return out
}

func TestISORandGolden(t *testing.T) {
	want := []uint32{
		11845, 16399, 13855, 21175, 18116, 11911, 15649, 26355,
		4292, 4791, 7, 32528, 26922, 4205, 16665, 24303,
	}
	if got := firstNISO(interleaveSeed, len(want)); !slices.Equal(got, want) {
		t.Fatalf("ISO Annex F sequence = %v, want %v", got, want)
	}
}

func TestISOLDPCRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		length int
		wc     int
		wr     int
	}{
		{580, 4, 9},
		{300, 4, 6},
	} {
		data := make([]byte, tc.length)
		for i := range data {
			data[i] = byte((i*7 + i/3) & 1)
		}
		codeword := EncodeLDPCProfile(data, tc.wc, tc.wr, wire.ISO23634)
		got, ok := DecodeLDPCHardProfile(codeword, tc.wc, tc.wr, wire.ISO23634)
		if !ok {
			t.Errorf("length %d weights (%d,%d): syndrome failed", tc.length, tc.wc, tc.wr)
			continue
		}
		if !slices.Equal(got, data) {
			t.Errorf("length %d weights (%d,%d): round trip differs", tc.length, tc.wc, tc.wr)
		}
	}
}

func TestISOInterleaveRoundTrip(t *testing.T) {
	want := make([]byte, 1044)
	for i := range want {
		want[i] = byte(i & 1)
	}
	got := slices.Clone(want)
	InterleaveProfile(got, wire.ISO23634)
	DeinterleaveProfile(got, wire.ISO23634)
	if !slices.Equal(got, want) {
		t.Fatal("ISO interleave round trip differs")
	}
}

func TestISOInterleaveGolden(t *testing.T) {
	in := make([]byte, 37)
	for i := range in {
		in[i] = byte(i*7 + 3)
	}
	want := []byte{
		136, 52, 227, 213, 17, 192, 66, 73, 255, 87, 234, 185, 45,
		122, 206, 38, 10, 59, 157, 108, 164, 115, 220, 199, 143, 178,
		3, 31, 24, 171, 241, 80, 248, 150, 101, 129, 94,
	}
	got := slices.Clone(in)
	InterleaveProfile(got, wire.ISO23634)
	if !slices.Equal(got, want) {
		t.Fatalf("ISO interleave = %v, want %v", got, want)
	}
}
