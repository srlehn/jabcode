// Package encode implements the JAB Code encoding pipeline: data analysis and
// bit-stream generation, LDPC and interleaving, module placement, masking, and
// bitmap rendering, for single- and multi-symbol codes. The public encoder
// package validates options and calls Run; the root package retains its common
// facade without owning a second implementation.
package encode

import (
	"errors"
	"fmt"
	"image"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

// Config is the resolved, validated encoder configuration the parent package
// passes to Run. Each slice is indexed by symbol, the primary symbol first.
type Config struct {
	Colors     int
	ModuleSize int
	ECCLevel   int // 0 means "default" (ECC level of the primary symbol)
	Format     wire.Encoding
	Opaque     bool // force exact byte-mode encoding and preserve the selected ECC

	SymbolNumber    int
	SymbolPositions []int
	SymbolVersions  []image.Point
	SymbolECCLevels []int
}

// encoder holds the mutable working state for one Run: the resolved
// configuration plus the palette, per-symbol matrices and the rendered bitmap.
type encoder struct {
	colors     int
	moduleSize int
	eccLevel   int
	format     wire.Encoding
	opaque     bool

	symbolNumber    int
	symbolPositions []int
	symbolVersions  []image.Point
	symbolECCLevels []int

	palette []byte
	symbols []symbol
	bitmap  *image.Paletted
}

// symbol is the internal per-symbol working state.
type symbol struct {
	index    int
	sideSize image.Point
	host     int
	docked   [4]int
	wcwr     [2]int
	data     []byte // bit-per-byte payload
	dataMap  []byte // 1 = data module, 0 = reserved (pattern/palette/metadata)
	metadata []byte
	matrix   []byte // module color indices
}

// Rendered is an encoded JAB Code together with the primary symbol's ground truth:
// its module matrix, side size and RGB palette. Detector diagnostics and the
// regression harness score against these.
type Rendered struct {
	Image    image.Image
	Matrix   []byte // primary symbol module color indices, row-major, SideSize.X wide
	SideSize image.Point
	Palette  []byte // packed RGB triples
}

// Run encodes data into a JAB Code image using the resolved configuration cfg.
// The caller is responsible for validating cfg.
func Run(cfg Config, data []byte) (image.Image, error) {
	r, err := Render(cfg, data)
	return r.Image, err
}

// Render encodes data like Run but also returns the primary symbol's rendered
// module matrix, side size and palette.
func Render(cfg Config, data []byte) (Rendered, error) {
	e := &encoder{
		colors:          cfg.Colors,
		moduleSize:      cfg.ModuleSize,
		eccLevel:        cfg.ECCLevel,
		format:          cfg.Format,
		opaque:          cfg.Opaque,
		symbolNumber:    cfg.SymbolNumber,
		symbolPositions: cfg.SymbolPositions,
		symbolVersions:  cfg.SymbolVersions,
		symbolECCLevels: cfg.SymbolECCLevels,
	}
	e.palette = palette.SetDefaultVariant(e.colors, e.format.Variant())
	if err := e.generate(data); err != nil {
		return Rendered{}, err
	}
	s := &e.symbols[0]
	return Rendered{Image: e.bitmap, Matrix: s.matrix, SideSize: s.sideSize, Palette: e.palette}, nil
}

// isDefaultMode reports whether the primary symbol can be encoded without
// explicit metadata.
func (e *encoder) isDefaultMode() bool {
	// Ports isDefaultMode in encoder.c.
	return e.colors == 8 && (e.eccLevel == 0 || e.eccLevel == spec.DefaultECCLevel)
}

// generate runs the encoding pipeline for a single primary symbol.
func (e *encoder) generate(data []byte) error {
	// Ports the single-symbol path of generateJABCode in encoder.c.
	if e.format == wire.EncodeBSI {
		return e.generateBSI(data)
	}
	if e.symbolNumber > 1 {
		return e.generateMulti(data)
	}
	e.symbols = []symbol{{index: 0, host: -1}}

	var encoded []byte
	if e.opaque {
		encoded = encodeOpaqueData(data)
	} else {
		seq, encodedLength := analyzeInputData(data)
		if seq == nil {
			return errEncode
		}
		var err error
		encoded, err = encodeData(data, encodedLength, seq)
		if err != nil {
			return err
		}
	}

	if err := e.setPrimarySymbolVersion(encoded); err != nil {
		return err
	}
	if err := e.fitDataIntoSymbol(encoded); err != nil {
		return err
	}
	if !e.isDefaultMode() {
		e.encodePrimaryMetadata()
	}

	s := &e.symbols[0]
	codeword := ecc.EncodeLDPCVariant(s.data, s.wcwr[0], s.wcwr[1], e.format.Variant())
	ecc.InterleaveVariant(codeword, e.format.Variant())
	e.createMatrix(0, codeword)

	// Default mode uses the fixed mask 7; otherwise pick the best mask and
	// re-encode the mask reference into the metadata.
	if e.isDefaultMode() {
		e.maskSymbol(0, spec.DefaultMaskingReference)
	} else {
		maskRef := e.maskCode(e.codePara())
		if maskRef != spec.DefaultMaskingReference {
			e.updatePrimaryMetadataPartII(maskRef)
			e.placePrimaryMetadataPartII()
		}
	}
	e.createBitmap()
	return nil
}

// symbolCapacity returns the data capacity in bits of a symbol of the given
// version.
func (e *encoder) symbolCapacity(version image.Point, primary bool) int {
	// Ports getSymbolCapacity in encoder.c.
	nbFinder := 4 * 7
	if primary {
		nbFinder = 4 * 17
	}
	palColors := min(e.colors, 64)
	// 4/8-color symbols carry the first two palette colors in the finder pattern;
	// the higher modes embed every color in the metadata region.
	nbPalette := (palColors - spec.PaletteFinderColors(e.colors)) * spec.PaletteCopies(e.colors)

	sx := spec.VersionToSize(version.X)
	sy := spec.VersionToSize(version.Y)
	apsX := tables.APNum[version.X-1]
	apsY := tables.APNum[version.Y-1]
	nbAlign := (apsX*apsY - 4) * 7

	bpm := spec.Log2Int(e.colors)
	nbMeta := 0
	if primary {
		metaBits := e.metadataLength()
		if metaBits > 0 {
			nbMeta = (metaBits - spec.PrimaryMetadataPart1Length) / bpm
			if (metaBits-spec.PrimaryMetadataPart1Length)%bpm != 0 {
				nbMeta++
			}
			nbMeta += spec.PrimaryMetadataPart1ModuleNumber
		}
	}
	return (sx*sy - nbFinder - nbAlign - nbPalette - nbMeta) * bpm
}

// metadataLength returns the encoded primary-symbol metadata bit length.
func (e *encoder) metadataLength() int {
	// Ports getMetadataLength for the primary symbol in encoder.c.
	if e.isDefaultMode() {
		return 0
	}
	return spec.PrimaryMetadataPart1Length + spec.PrimaryMetadataPart2Length
}

// netCapacity is the usable payload length after reserving LDPC parity, given a
// gross capacity and code-rate weights.
func netCapacity(capacity, wc, wr int) int {
	return (capacity/wr)*wr - (capacity/wr)*wc
}

// setPrimarySymbolVersion adopts an explicitly requested primary version, or
// picks the smallest square version that fits the payload.
func (e *encoder) setPrimarySymbolVersion(encoded []byte) error {
	// Ports the primary-symbol version selection in encoder.c.
	payloadLength := len(encoded) + 5 // plus S and flag bit
	if e.eccLevel == 0 {
		e.eccLevel = spec.DefaultECCLevel
	}
	s := &e.symbols[0]
	s.wcwr = [2]int{spec.ECCWeights[e.eccLevel][0], spec.ECCWeights[e.eccLevel][1]}

	// An explicitly requested version (possibly rectangular) is honoured as
	// given; fitDataIntoSymbol reports the error if the payload exceeds its
	// capacity. The reference generateJABCode auto-fits only "if not given".
	if len(e.symbolVersions) > 0 && e.symbolVersions[0] != (image.Point{}) {
		v := e.symbolVersions[0]
		if v.X < 1 || v.X > 32 || v.Y < 1 || v.Y > 32 {
			return fmt.Errorf("jabcode: incorrect symbol version %dx%d for the primary symbol", v.X, v.Y)
		}
		s.sideSize = image.Pt(spec.VersionToSize(v.X), spec.VersionToSize(v.Y))
		return nil
	}

	for v := 1; v <= 32; v++ {
		capacity := e.symbolCapacity(image.Pt(v, v), true)
		if netCapacity(capacity, s.wcwr[0], s.wcwr[1]) >= payloadLength {
			s.sideSize = image.Pt(spec.VersionToSize(v), spec.VersionToSize(v))
			return nil
		}
	}
	return errors.New("jabcode: message does not fit into one symbol; use more symbols")
}

// fitDataIntoSymbol builds the primary symbol's payload: the encoded message
// followed by the in-stream S metadata and flag bit, zero-padded to the net
// capacity.
func (e *encoder) fitDataIntoSymbol(encoded []byte) error {
	// Ports the default-mode single-symbol path of fitDataIntoSymbols in encoder.c.
	s := &e.symbols[0]
	version := image.Pt(spec.SizeToVersion(s.sideSize.X), spec.SizeToVersion(s.sideSize.Y))
	capacity := e.symbolCapacity(version, true)
	netCap := netCapacity(capacity, s.wcwr[0], s.wcwr[1])

	dataLen := len(encoded)
	payloadLen := dataLen + 1 + 4 // flag bit + primary S (4 bits)
	if payloadLen > netCap {
		return errors.New("jabcode: message does not fit; use a higher symbol version")
	}

	// Non-default symbols may pick a better code rate for the chosen version.
	pnLength := netCap
	if !e.isDefaultMode() && !e.opaque {
		optimalECC(capacity, payloadLen, &s.wcwr)
		pnLength = netCapacity(capacity, s.wcwr[0], s.wcwr[1])
	}

	s.data = make([]byte, pnLength)
	copy(s.data[:dataLen], encoded)
	pos := payloadLen - 1
	s.data[pos] = 1 // flag bit
	pos--
	for range 4 { // primary metadata S: no docked symbols -> 0
		s.data[pos] = 0
		pos--
	}
	return nil
}
