package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
)

// decodeSymbols finishes a read whose primary symbol is decoded in symbols[0]:
// it traverses every docked secondary once in breadth-first symbol order, then
// assembles and interprets the concatenated bit stream under the established
// wire variant.
func decodeSymbols(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, total int) (data []byte, ok bool) {
	return decodeSymbolsTraced(bm, ch, symbols, total, nil)
}

func decodeSymbolsTraced(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, total int, detail *DiagnosticAttempt) (data []byte, ok bool) {
	for i := 0; i < total && total < maxSymbolNumber; i++ {
		if !decodeDockedSecondariesTraced(bm, ch, symbols, i, &total, detail) {
			return nil, false
		}
	}

	// Concatenate the decoded bits of all symbols, then interpret them.
	n := 0
	for i := 0; i < total; i++ {
		n += len(symbols[i].Data)
	}
	bits := make([]byte, 0, n)
	for i := 0; i < total; i++ {
		bits = append(bits, symbols[i].Data...)
	}
	return decode.DecodeDataVariant(bits, symbols[0].WireVariant)
}

// decodeDockedSecondaries detects and decodes every secondary symbol docked to
// a host symbol. The host's established wire variant selects the one compiled
// physical-pattern and wire decoder needed by each secondary.
func decodeDockedSecondaries(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, hostIndex int, total *int) bool {
	return decodeDockedSecondariesTraced(bm, ch, symbols, hostIndex, total, nil)
}

func decodeDockedSecondariesTraced(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, hostIndex int, total *int, detail *DiagnosticAttempt) bool {
	// Ports decodeDockedSlaves in detector.c.
	dp := symbols[hostIndex].Meta.DockedPosition
	docked := [4]int{dp & 0x08, dp & 0x04, dp & 0x02, dp & 0x01}
	for dockedPosition := range 4 {
		if docked[dockedPosition] == 0 || *total >= maxSymbolNumber {
			continue
		}

		secondary := &symbols[*total]
		*secondary = core.DecodedSymbol{
			WireVariant: symbols[hostIndex].WireVariant,
			Index:       *total,
			HostIndex:   hostIndex,
			Meta:        symbols[hostIndex].SecondaryMeta[dockedPosition],
		}
		var classification decode.ModuleClassificationTrace
		var classificationTrace *decode.ModuleClassificationTrace
		if detail != nil {
			classificationTrace = &classification
		}
		matrix, metadataMatrix, result := decodeVariantDockedSecondary(
			bm, ch, &symbols[hostIndex], secondary, dockedPosition, classificationTrace,
		)
		appendDockedSecondaryTrace(detail, hostIndex, dockedPosition, secondary, matrix, metadataMatrix, result, classification)
		if result <= 0 {
			return false
		}
		(*total)++
	}
	return true
}

func decodeCurrentDockedSecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int, trace *decode.ModuleClassificationTrace) (*core.Bitmap, int) {
	matrix := detect.DetectSecondary(bm, ch, host, secondary, dockedPosition)
	if matrix == nil {
		return nil, core.Failure
	}
	if trace != nil {
		return matrix, decode.DecodeSecondaryTraced(matrix, secondary, trace)
	}
	return matrix, decode.DecodeSecondary(matrix, secondary)
}

func appendDockedSecondaryTrace(detail *DiagnosticAttempt, hostIndex, dockedPosition int, secondary *core.DecodedSymbol, matrix, metadataMatrix *core.Bitmap, result int, classification decode.ModuleClassificationTrace) {
	if detail == nil {
		return
	}
	secondaryTrace := DiagnosticSecondary{
		HostIndex: hostIndex, DockedPosition: dockedPosition, Matrix: matrix, MetadataMatrix: metadataMatrix,
		Result: result, Symbol: cloneDecodedSymbol(secondary), Classification: classification,
	}
	if matrix != nil {
		patterns := make([]detect.FinderPattern, 4)
		for i := range patterns {
			patterns[i] = detect.FinderPattern{
				Typ: i, Center: secondary.PatternPositions[i],
				ModuleSize: secondary.ModuleSize, FoundCount: 1,
			}
		}
		secondaryTrace.Side = secondary.SideSize
		secondaryTrace.Transform = core.PerspectiveTransform(
			patterns[0].Center, patterns[1].Center,
			patterns[2].Center, patterns[3].Center, secondary.SideSize,
		)
		secondaryTrace.HasTransform = true
		secondaryTrace.Patterns = patterns
	}
	detail.Secondaries = append(detail.Secondaries, secondaryTrace)
}
