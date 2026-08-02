package detect

import "testing"

// The rotated-decode gate cannot pin this: it only checks that the reported
// direction is one the search probes and that not every angle reports zero, so
// a constant 15, 45 or 75 would pass it. What has to hold is that the value is
// the direction actually marked published, which means direction zero must be
// reported as zero rather than confused with "nothing published", and the
// signature that was selected must be the one read.
func TestPublishedScanDegrees(t *testing.T) {
	scans := func(published int, degrees ...float64) []FinderFamilyScanStats {
		out := make([]FinderFamilyScanStats, len(degrees))
		for i, deg := range degrees {
			out[i] = FinderFamilyScanStats{Degrees: deg, Published: i == published}
		}
		return out
	}
	pass := func(family FinderFamily, published int, degrees ...float64) FinderPassStats {
		var p FinderPassStats
		p.familyStats(family).Scans = scans(published, degrees...)
		return p
	}

	t.Run("the published direction, not the last one", func(t *testing.T) {
		d := &PrimaryDetector{activeFamily: FinderFamilyCurrent, hasActiveFamily: true}
		d.Stats.Passes = []FinderPassStats{pass(FinderFamilyCurrent, 2, 0, 15, 30, 45)}
		if got := d.PublishedScanDegrees(); got != 30 {
			t.Fatalf("PublishedScanDegrees = %v, want 30", got)
		}
	})

	// Zero is the row walk and a real answer. Reporting it as "absent" is the
	// same confusion the route label used to carry.
	t.Run("the row walk reports zero", func(t *testing.T) {
		d := &PrimaryDetector{activeFamily: FinderFamilyCurrent, hasActiveFamily: true}
		d.Stats.Passes = []FinderPassStats{pass(FinderFamilyCurrent, 0, 0, 15, 30)}
		if got := d.PublishedScanDegrees(); got != 0 {
			t.Fatalf("PublishedScanDegrees = %v, want 0", got)
		}
	})

	t.Run("nothing published", func(t *testing.T) {
		d := &PrimaryDetector{activeFamily: FinderFamilyCurrent, hasActiveFamily: true}
		d.Stats.Passes = []FinderPassStats{pass(FinderFamilyCurrent, -1, 0, 15, 30)}
		if got := d.PublishedScanDegrees(); got != -1 {
			t.Fatalf("PublishedScanDegrees = %v, want -1", got)
		}
	})

	t.Run("no active family", func(t *testing.T) {
		d := &PrimaryDetector{}
		d.Stats.Passes = []FinderPassStats{pass(FinderFamilyCurrent, 1, 0, 45)}
		if got := d.PublishedScanDegrees(); got != -1 {
			t.Fatalf("PublishedScanDegrees = %v, want -1", got)
		}
	})

	// Passes accumulate and only the last one that published anything describes
	// the quad that was returned.
	t.Run("the latest pass that published", func(t *testing.T) {
		d := &PrimaryDetector{activeFamily: FinderFamilyCurrent, hasActiveFamily: true}
		d.Stats.Passes = []FinderPassStats{
			pass(FinderFamilyCurrent, 1, 0, 15, 30),
			pass(FinderFamilyCurrent, 3, 0, 15, 30, 75),
		}
		if got := d.PublishedScanDegrees(); got != 75 {
			t.Fatalf("PublishedScanDegrees = %v, want 75", got)
		}
	})

	// The two signatures share a pass and keep separate counters, so reading
	// the wrong one returns another family's direction.
	t.Run("the selected signature's own scans", func(t *testing.T) {
		if !bsiFamilyFinderEnabled {
			t.Skip("the BSI-era family is not compiled into this build")
		}
		d := &PrimaryDetector{activeFamily: FinderFamilyBSI, hasActiveFamily: true}
		p := pass(FinderFamilyCurrent, 1, 0, 15, 30)
		p.familyStats(FinderFamilyBSI).Scans = scans(2, 0, 30, 60)
		d.Stats.Passes = []FinderPassStats{p}
		if got := d.PublishedScanDegrees(); got != 60 {
			t.Fatalf("PublishedScanDegrees = %v, want the BSI family's 60", got)
		}
	})
}
