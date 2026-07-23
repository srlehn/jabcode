//go:build js

package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"syscall/js"
)

// webgpuFinderPassPreparer keeps retry binarization on the browser device.
// The detector still owns the small CPU-side average and pitch estimates, but
// adaptive-threshold and print-level mask generation use the same WGSL path
// as the initial resident pass.
type webgpuFinderPassPreparer struct {
	device *webgpuDevice
	bm     *core.Bitmap
	input  js.Value
	trace  bool
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
	if rx == 0 && ry == 0 {
		channels, err := preparer.device.webgpuBinarizeResident(
			preparer.input, preparer.bm, thresholds, printLevels,
		)
		return preparer.bm, channels, nil, nil, err
	}
	filtered, err := preparer.device.webgpuDescreenResident(
		preparer.input, preparer.bm.Width, preparer.bm.Height, rx, ry,
	)
	if err != nil {
		return nil, [3]*core.Bitmap{}, nil, nil, err
	}
	defer filtered.Call("destroy")
	channels, err := preparer.device.webgpuBinarizeResident(
		filtered, preparer.bm, thresholds, printLevels,
	)
	if err != nil || !preparer.trace {
		return preparer.bm, channels, nil, nil, err
	}
	imageBytes := len(preparer.bm.Pix)
	data, err := preparer.device.downloadBuffer(filtered, imageBytes)
	if err != nil {
		return nil, [3]*core.Bitmap{}, nil, nil, err
	}
	input := core.NewBitmap(preparer.bm.Width, preparer.bm.Height, preparer.bm.Channels)
	copy(input.Pix, data)
	return input, channels, nil, nil, nil
}
