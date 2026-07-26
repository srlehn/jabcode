package detect

import "testing"

// These pin the direction the two ambiguous side-size decisions resolve in.
// They assert the policy, not its correctness: the evidence for the policy is
// the version table and the docked perspective ceiling, and it is recorded in
// the plan. What they protect against is the direction being flipped back
// silently, since either decision reversed costs whole version steps on
// low-resolution frames and nothing else in the package would notice.
func TestSideSizeResolvesAmbiguityDownward(t *testing.T) {
	// Valid side sizes are 4*version+17, so a raw count of 3 mod 4 sits two
	// modules from either neighbour and the count alone cannot choose.
	for _, tc := range []struct{ raw, want, flag int }{
		{51, 49, 0},
		{55, 53, 0},
		{115, 113, 0},
		{49, 49, 1},
		{48, 49, 1},
		{50, 49, 1},
	} {
		got, flag := SideSize(tc.raw)
		if got != tc.want || flag != tc.flag {
			t.Errorf("SideSize(%d) = (%d, %d), want (%d, %d)", tc.raw, got, flag, tc.want, tc.flag)
		}
	}
}

func TestChooseAxisSizePrefersTheShorterEstimate(t *testing.T) {
	// Two edges equal on every quality signal: reliability, method agreement
	// and endpoint consistency. Only the sizes differ.
	a := edgeEstimate{size: 117, flag: 1, distSize: 121, distFlag: 1, msRatio: 1}
	b := edgeEstimate{size: 113, flag: 1, distSize: 117, distFlag: 1, msRatio: 1}
	if got := chooseAxisSize(a, b); got != 113 {
		t.Errorf("chooseAxisSize = %d, want the shorter estimate 113", got)
	}
	if got := chooseAxisSize(b, a); got != 113 {
		t.Errorf("chooseAxisSize is not symmetric in its arguments: %d", got)
	}
}
