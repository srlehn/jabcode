package jabcode

import (
	"errors"
	"fmt"
	"image"
	"math/bits"
)

// Encoding defaults and fixed sizes (jabcode.h, decoder.h).
const (
	defaultColorNumber      = 8
	defaultModuleSize       = 12
	defaultEccLevel         = 3
	defaultMaskingReference = 7

	colorPaletteNumber = 4
	distanceToBorder   = 4

	primaryMetadataX                 = 6
	primaryMetadataY                 = 1
	primaryMetadataPart1Length       = 6
	primaryMetadataPart2Length       = 38
	primaryMetadataPart1ModuleNumber = 4
)

// ecclevel2wcwr maps an error-correction level to its LDPC (wc, wr) weights
// (encoder.h).
var ecclevel2wcwr = [11][2]int{
	{4, 9}, {3, 8}, {3, 7}, {4, 9}, {3, 6}, {4, 7}, {4, 6}, {3, 4}, {4, 5}, {5, 6}, {6, 7},
}

func version2size(v int) int { return v*4 + 17 }
func size2version(s int) int { return (s - 17) / 4 }

// log2int returns log2(n) for a power-of-two n (the bits-per-module count).
func log2int(n int) int { return bits.Len(uint(n)) - 1 }

// symbol is the internal per-symbol working state (jab_symbol).
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

	// populated during Encode
	palette []byte
	symbols []symbol
	bitmap  *image.Paletted
}

// Option configures an Encoder.
type Option func(*Encoder)

// WithColors sets the number of module colors. Only 4 and 8 are supported, as in
// the reference jabcodeWriter; the default is 8.
func WithColors(n int) Option { return func(e *Encoder) { e.colors = n } }

// WithModuleSize sets the side length, in pixels, of each module.
func WithModuleSize(px int) Option { return func(e *Encoder) { e.moduleSize = px } }

// WithECCLevel sets the error-correction level (0..10); 0 selects the default.
func WithECCLevel(level int) Option { return func(e *Encoder) { e.eccLevel = level } }

// WithSymbols configures a multi-symbol code: one position (0..60), version
// (side-version x,y) and ECC level per symbol, the primary symbol first. For a
// slave symbol, ECC level 0 means "same as its host". (Multi-symbol uses
// non-default mode.)
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

// isDefaultMode reports whether the primary symbol can be encoded without
// explicit metadata (isDefaultMode in encoder.c).
func (e *Encoder) isDefaultMode() bool {
	return e.colors == 8 && (e.eccLevel == 0 || e.eccLevel == defaultEccLevel)
}

// validECCLevel reports whether an ECC level indexes a valid (wc, wr) pair.
func validECCLevel(level int) bool { return level >= 0 && level < len(ecclevel2wcwr) }

// validateSymbols checks the encoder configuration so that malformed options
// return an error instead of panicking later via table indexing.
func (e *Encoder) validateSymbols() error {
	if e.symbolNumber <= 1 {
		if !validECCLevel(e.eccLevel) {
			return fmt.Errorf("jabcode: invalid ECC level %d (valid: 0..%d)", e.eccLevel, len(ecclevel2wcwr)-1)
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
			return fmt.Errorf("jabcode: invalid ECC level %d for symbol %d (valid: 0..%d)", level, i, len(ecclevel2wcwr)-1)
		}
	}
	return nil
}

// Encode encodes data into a JAB Code image, single or multi-symbol, at any ECC
// level. It supports 4- and 8-color symbols, matching the reference jabcodeWriter
// (the library's >8-color palette placement is unverifiable — the reference tool
// itself only emits 4 and 8 colors — so it is rejected here).
func (e *Encoder) Encode(data []byte) (image.Image, error) {
	if !validColorNumber(e.colors) {
		return nil, fmt.Errorf("jabcode: invalid color number %d", e.colors)
	}
	if e.moduleSize < 1 {
		return nil, fmt.Errorf("jabcode: invalid module size %d", e.moduleSize)
	}
	if len(data) == 0 {
		return nil, errors.New("jabcode: no input data")
	}
	if e.colors != 4 && e.colors != 8 {
		return nil, fmt.Errorf("jabcode: only 4- and 8-color symbols are supported, not %d", e.colors)
	}
	if err := e.validateSymbols(); err != nil {
		return nil, err
	}

	e.palette = setDefaultPalette(e.colors)
	if err := e.generate(data); err != nil {
		return nil, err
	}
	return e.bitmap, nil
}

// generate runs the encoding pipeline for a single primary symbol
// (generateJABCode in encoder.c, single-symbol path).
func (e *Encoder) generate(data []byte) error {
	if e.symbolNumber > 1 {
		return e.generateMulti(data)
	}
	e.symbols = []symbol{{index: 0, host: -1}}

	seq, encodedLength := analyzeInputData(data)
	if seq == nil {
		return errEncode
	}
	encoded, err := encodeData(data, encodedLength, seq)
	if err != nil {
		return err
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
	ecc := encodeLDPC(s.data, s.wcwr[0], s.wcwr[1])
	interleaveData(ecc)
	e.createMatrix(0, ecc)

	// Default mode uses the fixed mask 7; otherwise pick the best mask and
	// re-encode the mask reference into the metadata.
	if e.isDefaultMode() {
		e.maskSymbol(0, defaultMaskingReference)
	} else {
		maskRef := e.maskCode(e.getCodePara())
		if maskRef != defaultMaskingReference {
			e.updatePrimaryMetadataPartII(maskRef)
			e.placePrimaryMetadataPartII()
		}
	}
	e.createBitmap()
	return nil
}

// symbolCapacity returns the data capacity in bits of a symbol of the given
// version (getSymbolCapacity in encoder.c).
func (e *Encoder) symbolCapacity(version image.Point, primary bool) int {
	nbFinder := 4 * 7
	if primary {
		nbFinder = 4 * 17
	}
	palColors := min(e.colors, 64)
	nbPalette := (palColors - 2) * colorPaletteNumber

	sx := version2size(version.X)
	sy := version2size(version.Y)
	apsX := apNum[version.X-1]
	apsY := apNum[version.Y-1]
	nbAlign := (apsX*apsY - 4) * 7

	bpm := log2int(e.colors)
	nbMeta := 0
	if primary {
		metaBits := e.metadataLength()
		if metaBits > 0 {
			nbMeta = (metaBits - primaryMetadataPart1Length) / bpm
			if (metaBits-primaryMetadataPart1Length)%bpm != 0 {
				nbMeta++
			}
			nbMeta += primaryMetadataPart1ModuleNumber
		}
	}
	return (sx*sy - nbFinder - nbAlign - nbPalette - nbMeta) * bpm
}

// metadataLength returns the encoded primary-symbol metadata bit length
// (getMetadataLength for the primary symbol).
func (e *Encoder) metadataLength() int {
	if e.isDefaultMode() {
		return 0
	}
	return primaryMetadataPart1Length + primaryMetadataPart2Length
}

// netCapacity is the usable payload length after reserving LDPC parity, given a
// gross capacity and code-rate weights.
func netCapacity(capacity, wc, wr int) int {
	return (capacity/wr)*wr - (capacity/wr)*wc
}

// setPrimarySymbolVersion picks the smallest square version that fits the
// payload (the primary-symbol version selection in encoder.c).
func (e *Encoder) setPrimarySymbolVersion(encoded []byte) error {
	payloadLength := len(encoded) + 5 // plus S and flag bit
	if e.eccLevel == 0 {
		e.eccLevel = defaultEccLevel
	}
	s := &e.symbols[0]
	s.wcwr = [2]int{ecclevel2wcwr[e.eccLevel][0], ecclevel2wcwr[e.eccLevel][1]}

	for v := 1; v <= 32; v++ {
		capacity := e.symbolCapacity(image.Pt(v, v), true)
		if netCapacity(capacity, s.wcwr[0], s.wcwr[1]) >= payloadLength {
			s.sideSize = image.Pt(version2size(v), version2size(v))
			return nil
		}
	}
	return errors.New("jabcode: message does not fit into one symbol; use more symbols")
}

// fitDataIntoSymbol builds the primary symbol's payload: the encoded message
// followed by the in-stream S metadata and flag bit, zero-padded to the net
// capacity (fitDataIntoSymbols in encoder.c, default-mode single-symbol path).
func (e *Encoder) fitDataIntoSymbol(encoded []byte) error {
	s := &e.symbols[0]
	version := image.Pt(size2version(s.sideSize.X), size2version(s.sideSize.Y))
	capacity := e.symbolCapacity(version, true)
	netCap := netCapacity(capacity, s.wcwr[0], s.wcwr[1])

	dataLen := len(encoded)
	payloadLen := dataLen + 1 + 4 // flag bit + primary S (4 bits)
	if payloadLen > netCap {
		return errors.New("jabcode: message does not fit; use a higher symbol version")
	}

	// Non-default symbols may pick a better code rate for the chosen version.
	pnLength := netCap
	if !e.isDefaultMode() {
		getOptimalECC(capacity, payloadLen, &s.wcwr)
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
