package read

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// DiagnosticTrace is the observation-only record of one authoritative Decode
// session. Attempts are ordered by the same deterministic route priority used
// to commit a decode result. Probes and ROI proposals are the actual search
// inputs used by those routes, not diagnostic recomputations.
type DiagnosticTrace struct {
	Input         image.Image
	Pyramid       []image.Point
	PyramidImages []image.Image
	Probes        []DiagnosticProbe
	ROIs          []DiagnosticROIs
	Attempts      []DiagnosticAttempt
}

// DiagnosticRoute identifies one concrete decode attempt.
type DiagnosticRoute struct {
	Kind  string
	Level int
	Angle float64
	ROI   int
}

// DiagnosticProbe records one orientation probe and the rungs it admitted.
type DiagnosticProbe struct {
	Level int
	ROI   int
	Label string
	Probe detect.CoarseProbeTrace
	Rungs []float64
}

// DiagnosticROIs records the actual ROI analysis used by one search route.
type DiagnosticROIs struct {
	Level      int
	Image      image.Image
	TileMap    detect.ROITileMap
	Candidates []detect.ROICandidate
}

// DiagnosticAttempt owns the intermediate state of one actual decode route.
// It is populated only for DecodeWithTrace; the normal Decode path passes nil
// and allocates none of this diagnostic state.
type DiagnosticAttempt struct {
	Route DiagnosticRoute
	Stage string

	Balanced        *core.Bitmap
	InitialChannels [3]*core.Bitmap
	FinalChannels   [3]*core.Bitmap
	Detector        detect.DetectorStats
	DetectorTrace   detect.DetectorTrace
	Finders         []detect.FinderPattern
	PrintDetected   bool

	Side           image.Point
	Transform      core.Perspective
	HasTransform   bool
	ChannelOffsets [3]core.PointF
	Sampled        *core.Bitmap
	Primary        []decode.PrimaryTrace
	Alignments     []*detect.AlignmentTrace
	Secondaries    []DiagnosticSecondary
	Payload        []byte
}

// DiagnosticSecondary records one docked-secondary sample and decode result.
type DiagnosticSecondary struct {
	HostIndex      int
	DockedPosition int
	Side           image.Point
	Transform      core.Perspective
	HasTransform   bool
	Patterns       []detect.FinderPattern
	Matrix         *core.Bitmap
	MetadataMatrix *core.Bitmap
	Symbol         core.DecodedSymbol
	Classification decode.ModuleClassificationTrace
	Result         int
}

// DecodeWithTrace runs the same decoder as Decode exactly once and returns its
// detailed observation trace. The trace cannot influence route selection or
// payload decisions.
func DecodeWithTrace(img image.Image) ([]byte, *DiagnosticTrace, error) {
	return DecodeWithTraceCapabilities(img, compiledCapabilities())
}

// DecodeWithTraceOnly is DecodeWithTrace under one selected internal variant.
func DecodeWithTraceOnly(img image.Image, variant wire.Variant) ([]byte, *DiagnosticTrace, error) {
	return DecodeWithTraceCapabilities(img, variant.Mask())
}

// DecodeWithTraceCapabilities is DecodeWithTrace with an additive decoder mask.
func DecodeWithTraceCapabilities(img image.Image, capabilities wire.Capabilities) ([]byte, *DiagnosticTrace, error) {
	tr := &routeTrace{level: -1, detailed: true}
	if err := validateCapabilities(capabilities); err != nil {
		return nil, &DiagnosticTrace{Input: img}, err
	}
	message, err := decodeRoutesCapabilities(img, tr, capabilities)
	return messageTransmission(message), &DiagnosticTrace{
		Input:         img,
		Pyramid:       append([]image.Point(nil), tr.pyramid...),
		PyramidImages: append([]image.Image(nil), tr.pyramidImages...),
		Probes:        append([]DiagnosticProbe(nil), tr.probes...),
		ROIs:          append([]DiagnosticROIs(nil), tr.rois...),
		Attempts:      append([]DiagnosticAttempt(nil), tr.details...),
	}, err
}

func cloneDecodedSymbol(s *core.DecodedSymbol) core.DecodedSymbol {
	if s == nil {
		return core.DecodedSymbol{}
	}
	out := *s
	out.Palette = append([]byte(nil), s.Palette...)
	out.Data = append([]byte(nil), s.Data...)
	return out
}
