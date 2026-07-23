//go:build js

package detect

import "github.com/srlehn/jabcode/internal/core"

// webgpuFinderPassPreparer keeps retry binarization on the browser device.
// The detector still owns the small CPU-side average and pitch estimates, but
// adaptive-threshold and print-level mask generation use the same WGSL path
// as the initial resident pass.
type webgpuFinderPassPreparer struct {
	device *webgpuDevice
	bm     *core.Bitmap
}

func (preparer webgpuFinderPassPreparer) averagePixelValue(fps []FinderPattern) ([3]float32, error) {
	return averagePixelValue(preparer.bm, fps), nil
}

func (preparer webgpuFinderPassPreparer) estimatePitch() (int, int, error) {
	px, py := EstimatePitch(preparer.bm)
	return px, py, nil
}

func (preparer webgpuFinderPassPreparer) prepare(
	rx, ry int,
	thresholds []float32,
	printLevels bool,
	_ uint32,
) (*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, func() error, error) {
	input := preparer.bm
	if rx > 0 || ry > 0 {
		input = descreen(input, rx, ry)
	}
	channels, err := preparer.device.webgpuBinarizeRGBWithThresholds(input, thresholds, printLevels)
	return input, channels, nil, nil, err
}
