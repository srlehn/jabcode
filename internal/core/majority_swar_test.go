//go:build !goexperiment.simd || !go1.27 || !(amd64 || arm64 || wasm)

package core

import "testing"

// TestMajority5MaskVotes checks the carry-save vote against a direct count on
// every one of the 32 input combinations. It carries the same build condition
// as the helper itself; the passes' equivalence tests deliberately do not, so
// they cover whichever kernel the build selected.
func TestMajority5MaskVotes(t *testing.T) {
	for pattern := range 32 {
		var in [5]uint64
		count := 0
		for k := range 5 {
			if pattern&(1<<k) != 0 {
				in[k] = ^uint64(0)
				count++
			}
		}
		want := uint64(0)
		if count > 2 {
			want = ^uint64(0)
		}
		if got := majority5Mask(in[0], in[1], in[2], in[3], in[4]); got != want {
			t.Fatalf("pattern %05b (count %d): got %#x, want %#x", pattern, count, got, want)
		}
	}
}
