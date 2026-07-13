package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/spec"
)

// PrimaryObservation is the observation half of a primary-symbol read: the
// sampled matrix with its metadata interpreted (part I with the
// default-metadata fallback, the embedded colour palette, part II) and every
// input the payload correction needs, before any data-module error
// correction has run. ObservePrimary produces it; CorrectPayload spends the
// correction. The split lets a caller hold a fully sampled and interpreted
// symbol, decide whether the expensive hard and soft LDPC chain is worth
// paying, and keep the observation for cross-frame use otherwise.
type PrimaryObservation struct {
	Matrix *core.Bitmap        // sampled module matrix
	Symbol *core.DecodedSymbol // metadata interpreted in place

	// PartISyndromeOK and PartIISyndromeOK record whether the hard-decoded
	// metadata parts satisfied their LDPC parity checks. They are recorded,
	// not enforced - metadata has fallback ladders of its own - and are
	// meaningful only when the parts actually decoded: a default-mode symbol
	// (Symbol.Meta.DefaultMode) decodes neither part.
	PartISyndromeOK  bool
	PartIISyndromeOK bool

	dataMap     []byte
	normPalette []float64
	palThs      []float64
	trace       *PrimaryTrace
}

// PrimaryTrace records the metadata, palette and payload-correction stages of
// one real primary observation. It is populated only by ObservePrimaryTraced.
type PrimaryTrace struct {
	Matrix *core.Bitmap

	PartIAttempted   bool
	PartIResult      int
	PartISyndromeOK  bool
	UsedDefault      bool
	PaletteAttempted bool
	PaletteResult    int
	PartIIAttempted  bool
	PartIIResult     int
	PartIISyndromeOK bool

	CorrectionAttempted bool
	CorrectionResult    int
	AdmissionChecked    bool
	Admitted            bool
	Classification      ModuleClassificationTrace
	Symbol              core.DecodedSymbol
}

func (tr *PrimaryTrace) capture(symbol *core.DecodedSymbol) {
	if tr == nil || symbol == nil {
		return
	}
	tr.Symbol = *symbol
	tr.Symbol.Palette = append([]byte(nil), symbol.Palette...)
	tr.Symbol.Data = append([]byte(nil), symbol.Data...)
}

// ObservePrimary interprets a sampled primary matrix up to but excluding
// payload correction. It returns the prepared observation on core.Success;
// nil and core.Failure when a metadata stage failed (symbol.Meta then holds
// whatever was interpreted - notably a part II side version that a caller's
// alignment-pattern retry may still use); nil and core.FatalError on a nil
// matrix.
func ObservePrimary(matrix *core.Bitmap, symbol *core.DecodedSymbol) (*PrimaryObservation, int) {
	return observePrimary(matrix, symbol, nil)
}

// ObservePrimaryTraced is ObservePrimary with stage results captured from the
// same execution.
func ObservePrimaryTraced(matrix *core.Bitmap, symbol *core.DecodedSymbol, trace *PrimaryTrace) (*PrimaryObservation, int) {
	return observePrimary(matrix, symbol, trace)
}

func observePrimary(matrix *core.Bitmap, symbol *core.DecodedSymbol, trace *PrimaryTrace) (*PrimaryObservation, int) {
	// Ports the metadata phase of decodePrimary in decoder.c.
	if trace != nil {
		*trace = PrimaryTrace{Matrix: matrix}
	}
	if matrix == nil {
		trace.capture(symbol)
		return nil, core.FatalError
	}
	if !spec.ValidSideSize(matrix.Width) || !spec.ValidSideSize(matrix.Height) {
		// A matrix that is no legal version size cannot be a JAB symbol, and
		// the metadata walks below assume at least the smallest legal side.
		// The internal samplers only produce legal sizes (the side-size
		// estimate snaps to 4v+17); this guards the seam for arbitrary
		// producers.
		trace.capture(symbol)
		return nil, core.Failure
	}
	symbol.SideSize = image.Pt(matrix.Width, matrix.Height)
	dataMap := make([]byte, matrix.Width*matrix.Height)

	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	moduleCount := 0

	partIRet, partISyn := DecodePrimaryMetadataPartI(matrix, symbol, dataMap, &moduleCount, &x, &y)
	if trace != nil {
		trace.PartIAttempted = true
		trace.PartIResult = partIRet
		trace.PartISyndromeOK = partISyn
	}
	if partIRet == core.Failure {
		trace.capture(symbol)
		return nil, core.Failure
	}
	if partIRet == MetadataFailed {
		x, y = spec.PrimaryMetadataX, spec.PrimaryMetadataY
		moduleCount = 0
		clear(dataMap)
		LoadDefaultPrimaryMetadata(matrix, symbol)
		if trace != nil {
			trace.UsedDefault = true
		}
	} else {
		// Part I decoded explicit metadata. Clear a default flag left by an
		// earlier observation of the same symbol (the alignment-pattern retry
		// re-observes in place), so the flag always describes THIS matrix.
		symbol.Meta.DefaultMode = false
	}

	paletteResult := ReadColorPaletteInPrimary(matrix, symbol, dataMap, &moduleCount, &x, &y)
	if trace != nil {
		trace.PaletteAttempted = true
		trace.PaletteResult = paletteResult
	}
	if paletteResult < 0 {
		trace.capture(symbol)
		return nil, core.Failure
	}

	colorNumber := 1 << (symbol.Meta.NC + 1)
	copies := spec.PaletteCopies(colorNumber)
	normPalette := make([]float64, colorNumber*4*copies)
	NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range copies {
		t := PaletteThreshold(symbol.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}

	partIISyn := false
	if partIRet == core.Success {
		var partIIRet int
		partIIRet, partIISyn = DecodePrimaryMetadataPartII(matrix, symbol, dataMap, normPalette, palThs, &moduleCount, &x, &y)
		if trace != nil {
			trace.PartIIAttempted = true
			trace.PartIIResult = partIIRet
			trace.PartIISyndromeOK = partIISyn
		}
		if partIIRet <= 0 {
			trace.capture(symbol)
			return nil, core.Failure
		}
	}
	trace.capture(symbol)

	return &PrimaryObservation{
		Matrix:           matrix,
		Symbol:           symbol,
		PartISyndromeOK:  partIRet == core.Success && partISyn,
		PartIISyndromeOK: partIISyn,
		dataMap:          dataMap,
		normPalette:      normPalette,
		palThs:           palThs,
		trace:            trace,
	}, core.Success
}

// CorrectPayload runs data-module error correction on the observed symbol -
// the expensive half of a primary read (demask, deinterleave, hard LDPC and
// the soft retry) - storing the net payload in the symbol's Data.
func (obs *PrimaryObservation) CorrectPayload() int {
	res := core.Failure
	if obs.trace != nil {
		res = DecodeSymbolTraced(obs.Matrix, obs.Symbol, obs.dataMap, obs.normPalette, obs.palThs, 0, &obs.trace.Classification)
	} else {
		res = DecodeSymbol(obs.Matrix, obs.Symbol, obs.dataMap, obs.normPalette, obs.palThs, 0)
	}
	if obs.trace != nil {
		obs.trace.CorrectionAttempted = true
		obs.trace.CorrectionResult = res
		obs.trace.capture(obs.Symbol)
	}
	return res
}
