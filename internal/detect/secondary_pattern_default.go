//go:build !jabcode_bsi && !jabcode_legacy

package detect

import "github.com/srlehn/jabcode/internal/core"

func findSecondaryAlignmentPattern(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int, family secondaryPatternFamily, b apBasis) FinderPattern {
	return findAlignmentPattern(ch, x, y, moduleSize, apType, b)
}
