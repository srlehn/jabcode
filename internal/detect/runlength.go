package detect

import (
	"encoding/binary"
	"math/bits"
)

// The finder cross-check walks spend nearly all their time inside runs of one
// value, where the walk only accumulates a length. Measuring a whole run at a
// time replaces the per-pixel branch, which is what actually costs there, with
// one word-at-a-time scan per run.
//
// Both helpers compare against a broadcast word rather than searching for the
// opposite value, so they hold for arbitrary byte content and do not depend on
// the channel bitmaps carrying exactly two values.

const broadcastByte = 0x0101010101010101

// leadingEqual reports how many leading bytes of s equal v.
func leadingEqual(s []byte, v byte) int {
	target := uint64(v) * broadcastByte
	i := 0
	for ; i+8 <= len(s); i += 8 {
		if x := binary.LittleEndian.Uint64(s[i:]) ^ target; x != 0 {
			return i + bits.TrailingZeros64(x)/8
		}
	}
	for ; i < len(s); i++ {
		if s[i] != v {
			return i
		}
	}
	return len(s)
}

// trailingEqual reports how many trailing bytes of s equal v.
func trailingEqual(s []byte, v byte) int {
	target := uint64(v) * broadcastByte
	n := len(s)
	i := 0
	for ; i+8 <= n; i += 8 {
		if x := binary.LittleEndian.Uint64(s[n-i-8:]) ^ target; x != 0 {
			// The differing byte nearest the end of this window ends the run;
			// every later byte in the window already matched.
			return i + 7 - (63-bits.LeadingZeros64(x))/8
		}
	}
	for ; i < n; i++ {
		if s[n-1-i] != v {
			return i
		}
	}
	return n
}
