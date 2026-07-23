package jabcode

import (
	"image"

	publicencoder "github.com/srlehn/jabcode/encoder"
)

// Encoder encodes data into a JAB Code. Configure it with the With* options;
// NewEncoder defaults to the experimental ISO/IEC 23634 format, 8 colors,
// module size 12 and the default ECC level.
type Encoder = publicencoder.Encoder

// Option configures an Encoder.
type Option = publicencoder.Option

// ControlKind identifies a structured encoder control.
type ControlKind = publicencoder.ControlKind

// Control places a structured encoder control relative to application data.
type Control = publicencoder.Control

const (
	ControlECI           = publicencoder.ControlECI
	ControlFNC1Start     = publicencoder.ControlFNC1Start
	ControlFNC1Separator = publicencoder.ControlFNC1Separator
	ControlFNC1End       = publicencoder.ControlFNC1End
)

// OpaquePlan is an immutable fixed-symbol byte-mode encoder plan.
type OpaquePlan = publicencoder.OpaquePlan

// WithColors sets the number of module colors.
//
// The default ISO encoder accepts 4 or 8 colors. More-than-8-color output
// requires jabcode_non_iso_encode and a non-ISO profile. Those denser modes
// have materially lower physical capture robustness; see encoder.WithColors
// for the measured limits.
func WithColors(n int) Option {
	return publicencoder.WithColors(n)
}

// WithModuleSize sets the side length, in pixels, of each module.
func WithModuleSize(px int) Option {
	return publicencoder.WithModuleSize(px)
}

// WithECCLevel sets the error-correction level (0..10); 0 selects the default.
func WithECCLevel(level int) Option {
	return publicencoder.WithECCLevel(level)
}

// WithControls adds structured ECI and FNC1 controls to the encoded message.
func WithControls(controls []Control) Option {
	return publicencoder.WithControls(controls)
}

// WithSymbols configures a fixed primary or a multi-symbol code. Each slice is
// indexed by symbol, with the primary first.
func WithSymbols(positions []int, versions []image.Point, eccLevels []int) Option {
	return publicencoder.WithSymbols(positions, versions, eccLevels)
}

// NewEncoder returns an Encoder configured by opts.
func NewEncoder(opts ...Option) *Encoder {
	return publicencoder.New(opts...)
}

// NewOpaquePlan creates a fixed single-symbol plan whose reported capacity is
// exact for arbitrary byte values.
func NewOpaquePlan(version image.Point, opts ...Option) (*OpaquePlan, error) {
	return publicencoder.NewOpaquePlan(version, opts...)
}
