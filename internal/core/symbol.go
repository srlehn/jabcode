package core

import "image"

// Status values shared by the detection and decoding stages.
const (
	Failure    = 0
	Success    = 1
	FatalError = -2
)

// Metadata holds a decoded symbol's parameters.
type Metadata struct {
	DefaultMode    bool
	NC             int // colour mode Nc; colour count = 2^(NC+1)
	MaskType       int
	DockedPosition int
	SideVersion    image.Point
	ECL            image.Point // error-correction (wc, wr)
}

// DecodedSymbol holds a decoded symbol.
type DecodedSymbol struct {
	Index            int
	HostIndex        int
	HostPosition     int
	SideSize         image.Point
	ModuleSize       float64
	PatternPositions [4]PointF
	Meta             Metadata
	SecondaryMeta    [4]Metadata
	Palette          []byte
	Data             []byte
}
