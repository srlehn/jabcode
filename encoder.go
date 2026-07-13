package jabcode

import (
	"errors"
	"fmt"
	"image"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
)

// Encoder defaults (jabcode.h).
const (
	defaultColorNumber = 8
	defaultModuleSize  = 12
	// maxSecondaryColors is the largest color count usable in a multi-symbol code.
	// A docked secondary places its palette at the fixed positions of a 32-entry
	// table, so it embeds at most 32 colors; a single symbol supports all counts.
	maxSecondaryColors = 32
)

// Encoder encodes data into a JAB Code. Configure it with the With* options;
// the zero defaults match the reference (8 colors, module size 12, default ECC).
type Encoder struct {
	colors      int
	moduleSize  int
	eccLevel    int // 0 means "default" (ECC level of the primary symbol)
	conformance ConformanceMode

	// Multi-symbol configuration (symbolNumber > 1). Each slice is indexed by
	// symbol, the primary symbol first.
	symbolNumber    int
	symbolPositions []int
	symbolVersions  []image.Point
	symbolECCLevels []int
}

// Option configures an Encoder.
type Option func(*Encoder)

// WithColors sets the number of module colors (4, 8, 16, 32, 64, 128 or 256); the
// default is 8.
//
// 4 and 8 are the interoperable modes: they match the reference jabcodeWriter and
// are read by other JAB Code software. 16 through 256 are a non-interoperable
// extension of this library - it encodes and decodes them, but no other decoder
// reads them: the reference implementation's normalized-RGB classifier cannot
// separate the intermediate color levels these palettes introduce, and it embeds
// the palette in four copies where the higher modes need the two-copy layout of
// ISO/IEC 23634 Annex G to fit the metadata region. This library follows Annex G
// for those modes (two embedded palettes; every color embedded up to 64, with
// 128/256 interpolated from that embedded 64) and classifies in absolute RGB.
//
// Physical robustness shrinks with the color count. Measured on real frontal,
// well-lit captures at the maximum ECC level: a phone camera photographing a
// display reads 16 colors reliably and 32 marginally, the same camera on a
// laser print reads up to 32, a flatbed scan reads up to 128, and 256 decodes
// only pixel-exact digital images. The measured limit is color classification,
// not geometry: past it, the per-module color error rate of a capture exceeds
// what the strongest error correction can repair (the packed palettes leave
// about one quantization step between neighbors, so camera noise and
// illumination casts collapse adjacent colors). Use the higher modes as a
// lossless digital container, or at most scanner-grade codes; prefer 4 or 8
// for anything camera-scanned or when the other end is not this library. A
// multi-symbol code caps at 32 colors (the secondary palette layout's limit).
func WithColors(n int) Option { return func(e *Encoder) { e.colors = n } }

// WithModuleSize sets the side length, in pixels, of each module.
func WithModuleSize(px int) Option { return func(e *Encoder) { e.moduleSize = px } }

// WithECCLevel sets the error-correction level (0..10); 0 selects the default.
func WithECCLevel(level int) Option { return func(e *Encoder) { e.eccLevel = level } }

// WithConformance selects C-reference compatibility or the ISO/IEC 23634:2022
// wire profile. C-reference compatibility is the default.
func WithConformance(mode ConformanceMode) Option {
	return func(e *Encoder) { e.conformance = mode }
}

// WithSymbols configures a multi-symbol code: one position (0..60), version
// (side-version x,y) and ECC level per symbol, the primary symbol first. For a
// slave symbol, ECC level 0 means "same as its host". (Multi-symbol uses
// non-default mode.) With a single entry it fixes the primary symbol's version
// explicitly - including rectangular ones - instead of auto-fitting the
// smallest square that holds the payload.
func WithSymbols(positions []int, versions []image.Point, eccLevels []int) Option {
	return func(e *Encoder) {
		e.symbolNumber = len(positions)
		e.symbolPositions = positions
		e.symbolVersions = versions
		e.symbolECCLevels = eccLevels
	}
}

// NewEncoder returns an Encoder configured by opts.
func NewEncoder(opts ...Option) *Encoder {
	e := &Encoder{colors: defaultColorNumber, moduleSize: defaultModuleSize, symbolNumber: 1}
	for _, o := range opts {
		o(e)
	}
	return e
}

func validColorNumber(n int) bool {
	switch n {
	case 4, 8, 16, 32, 64, 128, 256:
		return true
	}
	return false
}

// validECCLevel reports whether an ECC level indexes a valid (wc, wr) pair.
func validECCLevel(level int) bool { return level >= 0 && level < len(spec.ECCWeights) }

// validateSymbols checks the encoder configuration so that malformed options
// return an error instead of panicking later via table indexing.
func (e *Encoder) validateSymbols() error {
	if e.symbolNumber <= 1 {
		if !validECCLevel(e.eccLevel) {
			return fmt.Errorf("jabcode: invalid ECC level %d (valid: 0..%d)", e.eccLevel, len(spec.ECCWeights)-1)
		}
		if e.symbolPositions == nil && e.symbolVersions == nil && e.symbolECCLevels == nil {
			if e.symbolNumber == 0 {
				return errors.New("jabcode: WithSymbols needs at least one symbol")
			}
			return nil
		}
		// Any WithSymbols slice supplied means the explicit single-symbol
		// form; a partial call must not slip an unvalidated version or ECC
		// level through to the encoder.
		if len(e.symbolPositions) != 1 || e.symbolPositions[0] != 0 {
			return errors.New("jabcode: a single symbol must be at position 0")
		}
		if len(e.symbolVersions) != 1 || len(e.symbolECCLevels) != 1 {
			return errors.New("jabcode: WithSymbols needs one version and one ecc level for a single symbol")
		}
		if !validECCLevel(e.symbolECCLevels[0]) {
			return fmt.Errorf("jabcode: invalid ECC level %d for the primary symbol (valid: 0..%d)",
				e.symbolECCLevels[0], len(spec.ECCWeights)-1)
		}
		if e.eccLevel == 0 {
			e.eccLevel = e.symbolECCLevels[0]
		}
		return nil
	}
	if len(e.symbolPositions) != e.symbolNumber ||
		len(e.symbolVersions) != e.symbolNumber ||
		len(e.symbolECCLevels) != e.symbolNumber {
		return fmt.Errorf("jabcode: WithSymbols needs matching positions, versions and ecc levels (got %d, %d, %d)",
			len(e.symbolPositions), len(e.symbolVersions), len(e.symbolECCLevels))
	}
	for i, level := range e.symbolECCLevels {
		if !validECCLevel(level) {
			return fmt.Errorf("jabcode: invalid ECC level %d for symbol %d (valid: 0..%d)", level, i, len(spec.ECCWeights)-1)
		}
	}
	return nil
}

// Encode encodes data into a JAB Code image, single or multi-symbol, at any ECC
// level. A single symbol supports all color counts (4/8/16/32/64/128/256); 4 and
// 8 are interoperable with the reference jabcodeWriter, the higher modes are a
// non-interoperable extension only this library reads (see WithColors). A
// multi-symbol code caps at 32 colors, the limit of the secondary palette layout.
func (e *Encoder) Encode(data []byte) (image.Image, error) {
	if !validColorNumber(e.colors) {
		return nil, fmt.Errorf("jabcode: invalid color number %d", e.colors)
	}
	if !e.conformance.valid() {
		return nil, fmt.Errorf("jabcode: invalid conformance mode %d", e.conformance)
	}
	if e.conformance == ConformanceISO23634 && e.colors > 8 {
		return nil, fmt.Errorf("jabcode: ISO/IEC 23634 reserves module color modes above 8 colors")
	}
	if e.symbolNumber > 1 && e.colors > maxSecondaryColors {
		return nil, fmt.Errorf("jabcode: multi-symbol codes support at most %d colors, not %d (the docked-secondary palette layout has no positions beyond that)",
			maxSecondaryColors, e.colors)
	}
	if e.moduleSize < 1 {
		return nil, fmt.Errorf("jabcode: invalid module size %d", e.moduleSize)
	}
	if len(data) == 0 {
		return nil, errors.New("jabcode: no input data")
	}
	if err := e.validateSymbols(); err != nil {
		return nil, err
	}

	return encode.Run(encode.Config{
		Colors:          e.colors,
		ModuleSize:      e.moduleSize,
		ECCLevel:        e.eccLevel,
		Profile:         e.conformance.profile(),
		SymbolNumber:    e.symbolNumber,
		SymbolPositions: e.symbolPositions,
		SymbolVersions:  e.symbolVersions,
		SymbolECCLevels: e.symbolECCLevels,
	}, data)
}
