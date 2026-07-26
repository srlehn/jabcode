//go:build !jabcode_bsi && !jabcode_legacy

package detect

import "github.com/srlehn/jabcode/internal/core"

const bsiFamilyFinderEnabled = false

type optionalFinderPassStats struct{}

func (*FinderPassStats) startBSIFamily() {}

// BSIFamilyStats reports no optional signature in an untagged detector.
func (FinderPassStats) BSIFamilyStats() (FinderFamilyPassStats, bool) {
	return FinderFamilyPassStats{}, false
}

// familyStats has only the current signature's counters to return here; the BSI
// family never locates in an untagged build, so it never asks.
func (d *PrimaryDetector) familyStats(FinderFamily) *FinderFamilyPassStats {
	return &d.pass().FinderFamilyPassStats
}

func (*PrimaryDetector) scanBSIFamilyRow([3][]byte, int, *primaryFamilyScan) {}

func (*PrimaryDetector) scanDirectionalBSIFamily(scanDirection, int, *primaryFamilyScan) {}

func (*PrimaryDetector) consumeBSIFamilyHits(*finderPassRowHits, int, *primaryFamilyScan) {}

func (*PrimaryDetector) scanPatternVerticalBSIFamily(int, *primaryFamilyScan) {}

func (*PrimaryDetector) finishBSIFamilyScan(*primaryFamilyScan, float64) finderFamilyResult {
	return finderFamilyResult{status: core.Failure, scan: -1}
}
