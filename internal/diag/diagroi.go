package diag

import (
	"image"
	"io"

	"github.com/srlehn/jabcode/internal/detect"
)

// diagROIProposals dumps the ranked region-of-interest proposals for img, the
// measurement vehicle for whether the joint chroma/gradient score isolates the
// symbol before the proposer is wired into the decode path, and returns them for
// the follow-on per-region diagnostics.
func diagROIProposals(w io.Writer, img image.Image) []detect.ROICandidate {
	rois := detect.ProposeROIs(img, 6)
	diagLogf(w, "ROI proposals (joint chroma-variance x gradient-energy): %d", len(rois))
	for i, r := range rois {
		diagLogf(w, "  ROI %d score=%.3f chromaVar=%.3f grad=%.3f tiles=%d box=(%d,%d)-(%d,%d)",
			i, r.Score, r.ChromaVar, r.GradEnergy, r.Tiles,
			r.Bounds.Min.X, r.Bounds.Min.Y, r.Bounds.Max.X, r.Bounds.Max.Y)
	}
	diagROITileMap(w, img)
	return rois
}

// diagROITileMap prints the joint-score grid as an ASCII map banded relative to
// the peak tile, showing which tiles pass detect.ROIThreshold ('#') and how far
// below it the near-miss tiles sit - the evidence for whether a clipped symbol
// corner is a sub-threshold sliver (recoverable by a lower annex threshold) or
// scores zero.
func diagROITileMap(w io.Writer, img image.Image) {
	m := detect.BuildROITileMap(img)
	peak := m.Peak()
	if peak == 0 {
		diagLogf(w, "ROI tile map: flat image, no score peak")
		return
	}
	diagLogf(w, "ROI tile map (%dx%d tiles of %d work-px, work %dx%d, peak=%.4f):",
		m.GX, m.GY, m.Tile, m.W, m.H, peak)
	diagLogf(w, "  '#' >= %.2f*peak (ROIThreshold)  '+' >= %.2f (ROIAnnexThreshold)  '.' >= 0.01  ' ' below",
		detect.ROIThreshold, detect.ROIAnnexThreshold)
	for ty := range m.GY {
		row := make([]byte, m.GX)
		for tx := range m.GX {
			switch s := m.Score[ty*m.GX+tx] / peak; {
			case s >= detect.ROIThreshold:
				row[tx] = '#'
			case s >= detect.ROIAnnexThreshold:
				row[tx] = '+'
			case s >= 0.01:
				row[tx] = '.'
			default:
				row[tx] = ' '
			}
		}
		diagLogf(w, "  |%s|", row)
	}
}
