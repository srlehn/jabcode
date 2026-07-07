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
	// maxEncodableColors is the largest color count Encode accepts. Above it the
	// 64-entry embedded palette needs more placement modules than the primary
	// symbol's fixed metadata region holds, a structural limit of the format.
	maxEncodableColors = 32
)

// Encoder encodes data into a JAB Code. Configure it with the With* options;
// the zero defaults match the reference (8 colors, module size 12, default ECC).
type Encoder struct {
	colors     int
	moduleSize int
	eccLevel   int // 0 means "default" (ECC level of the primary symbol)

	// Multi-symbol configuration (symbolNumber > 1). Each slice is indexed by
	// symbol, the primary symbol first.
	symbolNumber    int
	symbolPositions []int
	symbolVersions  []image.Point
	symbolECCLevels []int
}

// Option configures an Encoder.
type Option func(*Encoder)

// WithColors sets the number of module colors; the default is 8.
//
// 4 and 8 are the interoperable modes: they match the reference jabcodeWriter and
// are read by other JAB Code software. 16 and 32 are a non-interoperable extension
// of this library - it encodes and decodes them, but no other decoder reads them
// (the reference detector is bound to the 8-color finder palette and its
// normalized-RGB classifier cannot separate the intermediate color levels). Prefer
// 4 or 8 unless both ends are this library. 64, 128 and 256 are rejected by Encode:
// their 64-entry palette overflows the primary symbol's fixed metadata capacity, a
// structural limit shared with the reference implementation.
func WithColors(n int) Option { return func(e *Encoder) { e.colors = n } }

// WithModuleSize sets the side length, in pixels, of each module.
func WithModuleSize(px int) Option { return func(e *Encoder) { e.moduleSize = px } }

// WithECCLevel sets the error-correction level (0..10); 0 selects the default.
func WithECCLevel(level int) Option { return func(e *Encoder) { e.eccLevel = level } }

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
// level. It supports 4-, 8-, 16- and 32-color symbols; 4 and 8 are interoperable
// with the reference jabcodeWriter, while 16 and 32 are a non-interoperable
// extension only this library reads (see WithColors). 64, 128 and 256 are rejected:
// their 64-entry palette overflows the primary symbol's fixed metadata capacity.
func (e *Encoder) Encode(data []byte) (image.Image, error) {
	if !validColorNumber(e.colors) {
		return nil, fmt.Errorf("jabcode: invalid color number %d", e.colors)
	}
	if e.colors > maxEncodableColors {
		return nil, fmt.Errorf("jabcode: %d-color symbols are not encodable: the %d-entry palette overflows the primary symbol's fixed metadata capacity (at most %d colors)",
			e.colors, e.colors, maxEncodableColors)
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
		SymbolNumber:    e.symbolNumber,
		SymbolPositions: e.symbolPositions,
		SymbolVersions:  e.symbolVersions,
		SymbolECCLevels: e.symbolECCLevels,
	}, data)
}
