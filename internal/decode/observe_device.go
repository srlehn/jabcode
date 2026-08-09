package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/spec"
)

// ObservePrimaryOnDevice interprets a sampled primary matrix from metadata a
// device already walked, and reports whether it answered at all.
//
// handled is false when the device declined - it does not own this sample, the
// colour mode is outside what it implements, or the symbol falls to the default
// metadata ladder, which reads the grid again from the start. The caller then
// runs ObservePrimary over the same matrix and nothing about the read changes.
//
// What comes back from a device is deliberately narrow: the colour mode, the
// palette bytes and the declared shape. The normalized palette, its thresholds
// and the reserved-module map are all rederived here, from those fields and the
// walk length, so every host stage downstream sees values its own arithmetic
// produced. A device that shipped its own normalized palette would be handing
// the host fallbacks a narrower float than they have ever classified against.
func ObservePrimaryOnDevice(
	device core.MetadataDevice,
	matrix *core.Bitmap,
	symbol *core.DecodedSymbol,
	trace *PrimaryTrace,
) (obs *PrimaryObservation, ret int, handled bool) {
	if device == nil || matrix == nil || symbol == nil {
		return nil, core.Failure, false
	}
	if !spec.ValidSideSize(matrix.Width) || !spec.ValidSideSize(matrix.Height) {
		return nil, core.Failure, false
	}
	meta, err := device.WalkPrimaryMetadata(matrix, symbol)
	if err != nil {
		return nil, core.Failure, false
	}
	colorNumber := 1 << (meta.NC + 1)
	copies := spec.PaletteCopies(colorNumber)
	// Checked before anything is written, so a decline leaves the symbol as the
	// host walk expects to find it.
	if meta.NC < 0 || len(meta.Palette) < colorNumber*3*copies {
		return nil, core.Failure, false
	}

	if trace != nil {
		*trace = PrimaryTrace{Matrix: matrix}
	}
	// A default-mode symbol carries no explicit shape, so the shape is the
	// format's constants and only the palette comes from the device. Setting it
	// here rather than trusting the record keeps one definition of what "default"
	// means, and it is the flag downstream stages branch on: the alignment path
	// confirms a side version only for a symbol in default mode.
	if meta.Defaulted {
		LoadDefaultPrimaryMetadata(matrix, symbol)
	} else {
		symbol.Meta.DefaultMode = false
		symbol.Meta.NC = meta.NC
		symbol.Meta.SideVersion = meta.SideVersion
		symbol.Meta.ECL = meta.ECL
		symbol.Meta.MaskType = meta.MaskType
		symbol.Meta.DockedPosition = 0
	}
	symbol.SideSize = image.Pt(matrix.Width, matrix.Height)
	symbol.Palette = meta.Palette

	dataMap := make([]byte, matrix.Width*matrix.Height)
	reserveMetadataModules(dataMap, matrix.Width, matrix.Height, meta.MetadataModules)
	if trace != nil {
		trace.PartIAttempted = true
		trace.PartIResult = core.Success
		trace.PartISyndromeOK = meta.PartISyndromeOK
		trace.PaletteAttempted = true
		// A default-mode symbol has no Part II to read, so the walk stopped after
		// the palette and the trace says so rather than claiming a stage that
		// never ran.
		trace.UsedDefault = meta.Defaulted
		if meta.Defaulted {
			trace.PartIResult = MetadataFailed
		}
		trace.PartIIAttempted = !meta.Defaulted
		trace.PartIISyndromeOK = meta.PartIISyndromeOK
		// The device walks the three stages without stopping between them, so
		// the per-stage maps the diagnostics draw are replayed here instead.
		// Each stage's length follows from the colour mode alone, which is why
		// no part of this had to come back with the record.
		trace.PartIDataMap = replayMetadataModules(matrix, spec.PrimaryMetadataPart1ModuleNumber)
		trace.PaletteDataMap = replayMetadataModules(matrix,
			spec.PrimaryMetadataPart1ModuleNumber+(colorNumber-2)*spec.ColorPaletteNumber)
		trace.PartIIDataMap = append([]byte(nil), dataMap...)
	}

	if meta.Rejected {
		// The declared shape contradicts the sample. The host walk reports this
		// as a failed observation that still carries the version it read,
		// because the caller's alignment-pattern retry resamples at that
		// version; a rejection that cleared the metadata would strand it.
		symbol.SideSize = image.Pt(
			spec.VersionToSize(meta.SideVersion.X), spec.VersionToSize(meta.SideVersion.Y))
		if trace != nil {
			trace.PartIIResult = core.Failure
		}
		trace.capture(symbol)
		return nil, core.Failure, true
	}
	if trace != nil && !meta.Defaulted {
		trace.PartIIResult = core.Success
	}

	normPalette := make([]float64, colorNumber*4*copies)
	NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range copies {
		t := PaletteThreshold(symbol.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}
	trace.capture(symbol)

	return &PrimaryObservation{
		Matrix:           matrix,
		Symbol:           symbol,
		PartISyndromeOK:  meta.PartISyndromeOK,
		PartIISyndromeOK: meta.PartIISyndromeOK,
		dataMap:          dataMap,
		normPalette:      normPalette,
		palThs:           palThs,
		trace:            trace,
		metaModules:      meta.MetadataModules,
	}, core.Success, true
}

func replayMetadataModules(matrix *core.Bitmap, count int) []byte {
	dataMap := make([]byte, matrix.Width*matrix.Height)
	reserveMetadataModules(dataMap, matrix.Width, matrix.Height, count)
	return dataMap
}

// reserveMetadataModules marks the modules a primary metadata walk of the given
// length consumes. The walk's path depends on the symbol's side and its own
// step count and not at all on what the modules hold, which is why a host that
// never read them can still reserve exactly the ones that were read.
func reserveMetadataModules(dataMap []byte, width, height, count int) {
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	for i := range count {
		if x < 0 || y < 0 || x >= width || y >= height {
			return
		}
		dataMap[y*width+x] = 1
		spec.NextMetadataModuleInPrimary(height, width, i+1, &x, &y)
	}
}
