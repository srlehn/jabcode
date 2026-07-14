package read

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
)

const maxAlignmentSamples = 2

type alignmentSampleCache struct {
	entries [maxAlignmentSamples]alignmentSample
	count   int
}

type alignmentSample struct {
	inputVersion  image.Point
	inputSide     image.Point
	defaultMode   bool
	outputVersion image.Point
	outputSide    image.Point
	matrix        *core.Bitmap
	trace         *detect.AlignmentTrace
}

func (cache *alignmentSampleCache) find(symbol *core.DecodedSymbol) *alignmentSample {
	if cache == nil {
		return nil
	}
	for i := range cache.count {
		entry := &cache.entries[i]
		if entry.inputVersion == symbol.Meta.SideVersion && entry.inputSide == symbol.SideSize && entry.defaultMode == symbol.Meta.DefaultMode {
			return entry
		}
	}
	return nil
}

func (cache *alignmentSampleCache) add(inputVersion, inputSide image.Point, defaultMode bool, symbol *core.DecodedSymbol, matrix *core.Bitmap, trace *detect.AlignmentTrace) {
	if cache == nil || cache.count >= len(cache.entries) {
		return
	}
	cache.entries[cache.count] = alignmentSample{
		inputVersion:  inputVersion,
		inputSide:     inputSide,
		defaultMode:   defaultMode,
		outputVersion: symbol.Meta.SideVersion,
		outputSide:    symbol.SideSize,
		matrix:        matrix,
		trace:         trace,
	}
	cache.count++
}

// samplePrimaryByAlignment reuses an actual alignment-pattern sample only
// when the interpreted input geometry is identical. Different versions or
// default-mode size confirmation get distinct authoritative samples and trace
// entries.
func samplePrimaryByAlignment(bm *core.Bitmap, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []detect.FinderPattern, detail *DiagnosticAttempt, cache *alignmentSampleCache) *core.Bitmap {
	if entry := cache.find(symbol); entry != nil {
		symbol.Meta.SideVersion = entry.outputVersion
		symbol.SideSize = entry.outputSide
		if entry.trace != nil {
			entry.trace.ReuseCount++
		}
		return entry.matrix
	}

	inputVersion := symbol.Meta.SideVersion
	inputSide := symbol.SideSize
	defaultMode := symbol.Meta.DefaultMode
	var trace *detect.AlignmentTrace
	if detail != nil {
		trace = &detect.AlignmentTrace{}
		detail.Alignments = append(detail.Alignments, trace)
	}
	var matrix *core.Bitmap
	if trace != nil {
		matrix = detect.SampleSymbolByAlignmentPatternTraced(bm, ch, symbol, fps, trace)
	} else {
		matrix = detect.SampleSymbolByAlignmentPattern(bm, ch, symbol, fps)
	}
	cache.add(inputVersion, inputSide, defaultMode, symbol, matrix, trace)
	return matrix
}
