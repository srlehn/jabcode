//go:build jabcode_bsi || jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// decodeHistoricalLocated samples the finder family shared by BSI TR-03137 and
// the pre-v2.0 C reference from the shared detector traversal once, then tries
// every enabled interpretation requested by capabilities.
func decodeHistoricalLocated(d *detect.PrimaryDetector, f *finding, detail *DiagnosticAttempt, capabilities wire.Capabilities) ([]byte, readStage, bool) {
	if !d.SelectFinderFamily(detect.FinderFamilyBSI) {
		return nil, readNoFinders, finderEvidence(d)
	}
	bm, ch := d.BM, d.Ch
	fps := d.FPs
	side := detect.CalculateSideSize(bm, fps)
	if side.X == -1 || side.Y == -1 {
		if quad, ok := d.SelectFinderQuadByGeometry(); ok {
			copy(fps, quad[:])
			side = detect.CalculateSideSize(bm, fps)
		}
		if side.X == -1 || side.Y == -1 {
			return nil, readNoSideSize, true
		}
	}
	if f != nil {
		for i := range 4 {
			f.quad[i] = fps[i].Center
			f.sizes[i] = fps[i].ModuleSize
		}
		f.side = side
		f.family = detect.FinderFamilyBSI
		f.located = true
	}

	transform := core.PerspectiveTransform(fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, side)
	matrix := detect.SampleSymbol(bm, transform, side)
	if matrix == nil {
		return nil, readNoSample, true
	}
	if detail != nil {
		detail.Side = side
		detail.Transform = transform
		detail.HasTransform = true
		detail.Sampled = matrix
		detail.FinalChannels = d.Ch
		detail.Detector = d.Stats
		detail.Finders = append([]detect.FinderPattern(nil), fps[:4]...)
	}

	base := core.DecodedSymbol{
		Index:      0,
		HostIndex:  0,
		SideSize:   side,
		ModuleSize: (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4,
		PatternPositions: [4]core.PointF{
			fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center,
		},
	}
	if capabilities.Has(wire.BSI) {
		if data, ok := decodeBSISampled(matrix, base); ok {
			if f != nil && f.located {
				f.payload = data
			}
			return data, readDecoded, true
		}
	}
	if capabilities.Has(wire.PreV2C) {
		if data, ok := decodePreV2CSampled(bm, ch, matrix, base, detail); ok {
			if f != nil && f.located {
				f.payload = data
			}
			return data, readDecoded, true
		}
	}
	return nil, readSampled, true
}
