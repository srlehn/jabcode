package ecc

import (
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

// Golden vectors captured from the C reference (libjabcode.a) via a small
// harness. They pin the LCG output and the (de)interleave permutation so any
// divergence from the reference is caught.

func firstN(seed uint64, n int) []uint32 {
	out := make([]uint32, 0, n)
	for x := range lcgValues(seed) {
		out = append(out, x)
		if len(out) == n {
			break
		}
	}
	return out
}

func TestLCGGolden(t *testing.T) {
	cases := []struct {
		seed uint64
		want []uint32
	}{
		{42, []uint32{716946689, 2976224391, 2616661493, 2496602140, 608325226, 21786605, 1531751673, 3169952270, 990497055, 2674544133, 1893180869, 2812421028, 6045153, 765986520, 1472052562, 2495308764}},
		{226759, []uint32{3605414883, 3579144034, 3982877425, 87907282, 3103878523, 256198050, 3255258821, 717530923, 623558569, 602349096, 5623112, 312490180, 2983108645, 4191715673, 87214728, 315541496}},
	}
	for _, c := range cases {
		got := firstN(c.seed, len(c.want))
		for i, want := range c.want {
			if got[i] != want {
				t.Fatalf("seed %d: value[%d] = %d, want %d", c.seed, i, got[i], want)
			}
		}
	}
}

func TestInterleaveGolden(t *testing.T) {
	in := make([]byte, 37)
	for i := range in {
		in[i] = byte(i*7 + 3)
	}
	want := []byte{101, 150, 143, 87, 129, 157, 171, 45, 52, 66, 59, 206, 108, 192, 115, 73, 136, 80, 17, 94, 199, 178, 185, 248, 122, 255, 234, 24, 31, 38, 241, 10, 164, 3, 227, 213, 220}

	data := append([]byte(nil), in...)
	InterleaveProfile(data, wire.Legacy)
	for i := range want {
		if data[i] != want[i] {
			t.Fatalf("interleave[%d] = %d, want %d", i, data[i], want[i])
		}
	}

	DeinterleaveProfile(data, wire.Legacy)
	for i := range in {
		if data[i] != in[i] {
			t.Fatalf("deinterleave[%d] = %d, want %d (round-trip failed)", i, data[i], in[i])
		}
	}
}
