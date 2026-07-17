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

func (*PrimaryDetector) scanBSIFamilyRow([3][]byte, int, *primaryFamilyScan) {}

func (*PrimaryDetector) consumeBSIFamilyHits([]finderRowHit, int, *primaryFamilyScan) {}

func (*PrimaryDetector) scanPatternVerticalBSIFamily(int, *primaryFamilyScan) {}

func (*PrimaryDetector) finishBSIFamilyScan(*primaryFamilyScan) finderFamilyResult {
	return finderFamilyResult{status: core.Failure}
}
