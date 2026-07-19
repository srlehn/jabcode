//go:build js

package detect

import (
	"errors"
	"image"

	"github.com/srlehn/jabcode/internal/core"
)

var errGPUDecodeUnavailable = errors.New("jabcode: GPU decode is unavailable on JavaScript targets")

// GPUDecodeSession preserves the common reader surface on JavaScript targets. The
// automatic constructor always returns nil so the existing CPU pipeline runs.
type GPUDecodeSession struct{}

// NewAutomaticGPUDecodeSession selects the CPU reader on JavaScript targets.
func NewAutomaticGPUDecodeSession(*core.Bitmap, int) (*GPUDecodeSession, error) {
	return nil, nil
}

// DownloadLevel reports that no resident GPU pyramid exists on JavaScript targets.
func (*GPUDecodeSession) DownloadLevel(int) (*core.Bitmap, error) {
	return nil, errGPUDecodeUnavailable
}

// LocateLevelFamilies reports that the caller should use its CPU fallback.
func (*GPUDecodeSession) LocateLevelFamilies(
	int,
	FinderFamilySet,
	int,
	func() bool,
	*DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	return nil, 0, errGPUDecodeUnavailable
}

// LocateRouteFamilies reports that the caller should use its CPU fallback.
func (*GPUDecodeSession) LocateRouteFamilies(
	int,
	image.Rectangle,
	float64,
	FinderFamilySet,
	int,
	func() bool,
	*DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, image.Point, error) {
	return nil, 0, image.Point{}, errGPUDecodeUnavailable
}

// ProbeLevelFamilies reports that the coarse probe was not handled.
func (*GPUDecodeSession) ProbeLevelFamilies(int, *CoarseProbeTrace) ([]CoarseFamily, bool) {
	return nil, false
}

// Close is safe for the no-GPU JavaScript session surface.
func (*GPUDecodeSession) Close() error {
	return nil
}
