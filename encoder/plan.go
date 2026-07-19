package encoder

import (
	"errors"
	"fmt"
	"image"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

// OpaquePlan is an immutable, fixed-primary-symbol encoder plan for arbitrary
// bytes. It uses byte mode for the entire message, so Capacity is independent
// of the byte values and every accepted payload has identical geometry.
type OpaquePlan struct {
	cfg        encode.Config
	version    image.Point
	modules    image.Point
	pixels     image.Point
	capacity   int
	colors     int
	eccLevel   int
	moduleSize int
}

// NewOpaquePlan creates a fixed single-symbol byte-mode plan. version contains
// the horizontal and vertical side versions, each in the range 1..32. Options
// may select colors, module size, ECC and an ISO-family profile, but must not
// configure symbols. ECC level 0 resolves to the standard default level.
// BSI and multi-symbol output are not opaque-plan formats.
func NewOpaquePlan(version image.Point, opts ...Option) (*OpaquePlan, error) {
	e := New(opts...)
	if e.symbolNumber != 1 || e.symbolPositions != nil || e.symbolVersions != nil || e.symbolECCLevels != nil {
		return nil, errors.New("jabcode: an opaque plan owns its fixed primary symbol; do not use WithSymbols")
	}
	if err := e.validate(); err != nil {
		return nil, err
	}
	if version.X < 1 || version.X > 32 || version.Y < 1 || version.Y > 32 {
		return nil, fmt.Errorf("jabcode: invalid opaque-plan version %dx%d (valid: 1..32 per side)", version.X, version.Y)
	}
	if e.format != wire.EncodeISO23634 && e.format != wire.EncodeISOHighColor {
		return nil, errors.New("jabcode: opaque plans require an ISO-family encoding profile")
	}

	eccLevel := e.eccLevel
	if eccLevel == 0 {
		eccLevel = spec.DefaultECCLevel
	}
	cfg := encode.Config{
		Colors:          e.colors,
		ModuleSize:      e.moduleSize,
		ECCLevel:        eccLevel,
		Format:          e.format,
		Opaque:          true,
		SymbolNumber:    1,
		SymbolPositions: []int{0},
		SymbolVersions:  []image.Point{version},
		SymbolECCLevels: []int{eccLevel},
	}
	capacity, err := encode.OpaqueCapacity(cfg)
	if err != nil {
		return nil, err
	}
	modules := image.Pt(spec.VersionToSize(version.X), spec.VersionToSize(version.Y))
	return &OpaquePlan{
		cfg: cfg, version: version, modules: modules,
		pixels:   image.Pt(modules.X*e.moduleSize, modules.Y*e.moduleSize),
		capacity: capacity, colors: e.colors, eccLevel: eccLevel, moduleSize: e.moduleSize,
	}, nil
}

// Colors returns the fixed number of module colors.
func (p *OpaquePlan) Colors() int { return p.colors }

// Version returns the fixed horizontal and vertical side versions.
func (p *OpaquePlan) Version() image.Point { return p.version }

// ECCLevel returns the resolved, fixed error-correction level.
func (p *OpaquePlan) ECCLevel() int { return p.eccLevel }

// ModuleSize returns the side length of one rendered module in pixels.
func (p *OpaquePlan) ModuleSize() int { return p.moduleSize }

// ModuleDimensions returns the fixed symbol dimensions in modules.
func (p *OpaquePlan) ModuleDimensions() image.Point { return p.modules }

// ImageDimensions returns the fixed rendered dimensions in pixels.
func (p *OpaquePlan) ImageDimensions() image.Point { return p.pixels }

// Capacity returns the exact number of arbitrary bytes that fit this plan.
func (p *OpaquePlan) Capacity() int { return p.capacity }

// Encode renders data according to the fixed plan. Empty messages are rejected
// explicitly. A payload longer than Capacity returns an error without invoking
// the encoder.
func (p *OpaquePlan) Encode(data []byte) (image.Image, error) {
	if p == nil {
		return nil, errors.New("jabcode: nil opaque plan")
	}
	if len(data) == 0 {
		return nil, errors.New("jabcode: opaque plans do not encode empty payloads")
	}
	if len(data) > p.capacity {
		return nil, fmt.Errorf("jabcode: opaque payload is %d bytes; fixed-symbol capacity is %d", len(data), p.capacity)
	}
	return encode.Run(p.cfg, data)
}
