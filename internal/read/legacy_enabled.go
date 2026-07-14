//go:build jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

const legacyReadEnabled = true

func decodeLegacySampled(bm *core.Bitmap, ch [3]*core.Bitmap, matrix *core.Bitmap, base core.DecodedSymbol, detail *DiagnosticAttempt) ([]byte, bool) {
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbols[0] = base
	symbols[0].WireProfile = wire.Legacy
	if decode.DecodeLegacyPrimary(matrix, &symbols[0]) != core.Success {
		return nil, false
	}
	return decodeLegacySymbolsTraced(bm, ch, symbols, 1, detail)
}

// decodeLegacySymbolsTraced follows every secondary attached to a legacy JAB
// Code symbol emitted by the pre-v2.0 C reference implementation, then decodes
// the assembled message using that implementation's wire profile.
func decodeLegacySymbolsTraced(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, total int, detail *DiagnosticAttempt) ([]byte, bool) {
	for i := 0; i < total && total < maxSymbolNumber; i++ {
		if !decodeLegacyDockedSecondariesTraced(bm, ch, symbols, i, &total, detail) {
			return nil, false
		}
	}

	n := 0
	for i := 0; i < total; i++ {
		n += len(symbols[i].Data)
	}
	bits := make([]byte, 0, n)
	for i := 0; i < total; i++ {
		bits = append(bits, symbols[i].Data...)
	}
	return decode.DecodeDataProfile(bits, wire.CReference)
}

func decodeLegacyDockedSecondariesTraced(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, hostIndex int, total *int, detail *DiagnosticAttempt) bool {
	dp := symbols[hostIndex].Meta.DockedPosition
	docked := [4]int{dp & 0x08, dp & 0x04, dp & 0x02, dp & 0x01}
	for j := range 4 {
		if docked[j] == 0 || *total >= maxSymbolNumber {
			continue
		}

		secondary := &symbols[*total]
		secondary.WireProfile = wire.CReference
		secondary.Index = *total
		secondary.HostIndex = hostIndex
		secondary.Meta = symbols[hostIndex].SecondaryMeta[j]
		matrix := detect.DetectLegacySecondary(bm, ch, &symbols[hostIndex], secondary, j)
		if matrix == nil {
			if detail != nil {
				detail.Secondaries = append(detail.Secondaries, DiagnosticSecondary{
					HostIndex: hostIndex, DockedPosition: j, Result: core.Failure,
					Symbol: cloneDecodedSymbol(secondary),
				})
			}
			return false
		}

		result := decode.DecodeLegacySecondary(matrix, secondary)
		if detail != nil {
			patterns := make([]detect.FinderPattern, 4)
			for i := range patterns {
				patterns[i] = detect.FinderPattern{
					Typ: i, Center: secondary.PatternPositions[i],
					ModuleSize: secondary.ModuleSize, FoundCount: 1,
				}
			}
			pt := core.PerspectiveTransform(
				patterns[0].Center, patterns[1].Center,
				patterns[2].Center, patterns[3].Center,
				secondary.SideSize,
			)
			detail.Secondaries = append(detail.Secondaries, DiagnosticSecondary{
				HostIndex: hostIndex, DockedPosition: j,
				Side: secondary.SideSize, Transform: pt, HasTransform: true,
				Patterns: patterns, Matrix: matrix, Result: result,
				Symbol: cloneDecodedSymbol(secondary),
			})
		}
		if result != core.Success {
			return false
		}
		*total++
	}
	return true
}
