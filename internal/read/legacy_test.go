package read

import (
	"image"
	"image/draw"
	"image/png"
	"os"
	"testing"

	"github.com/srlehn/jabcode/internal/testutil"
)

func loadLegacyCReferenceFixture(t *testing.T, name string) image.Image {
	t.Helper()
	f, err := os.Open(testutil.TestdataPath(name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func testNRGBA(img image.Image) *image.NRGBA {
	bounds := img.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(dst, dst.Bounds(), img, bounds.Min, draw.Src)
	return dst
}
