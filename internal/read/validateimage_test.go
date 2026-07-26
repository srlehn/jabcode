package read

import (
	"errors"
	"image"
	"math"
	"testing"
)

// The public entry points recover panics into a generic error, so a test that
// only asserts "some error came back" cannot tell validation from a rescued
// out-of-range read. These call validateImage directly, which is the boundary
// the raster checks actually own.

func TestValidateImageRejectsOverflowingStride(t *testing.T) {
	// (Dy-1)*Stride wraps to zero, so the first and last rows land inside Pix
	// while every row between them is far outside it. Expressed from MaxInt so
	// the wrap happens at either int width.
	img := &image.NRGBA{
		Pix:    make([]byte, 8),
		Stride: (math.MaxInt >> 1) + 1,
		Rect:   image.Rect(0, 0, 2, 5),
	}
	if err := validateImage(img); !errors.Is(err, errInvalidImage) {
		t.Fatalf("accepted a raster whose row arithmetic overflows: %v", err)
	}
}

func TestValidateImageRejectsShortNYCbCrAAlpha(t *testing.T) {
	// Luma and chroma are consistent; only the alpha plane is short. NYCbCrA is
	// classified safe for concurrent reads and AOffset bounds-checks nothing, so
	// an unvalidated alpha plane is a worker-goroutine panic.
	img := &image.NYCbCrA{
		YCbCr: image.YCbCr{
			Y: make([]byte, 4), Cb: make([]byte, 4), Cr: make([]byte, 4),
			YStride: 2, CStride: 2,
			SubsampleRatio: image.YCbCrSubsampleRatio444,
			Rect:           image.Rect(0, 0, 2, 2),
		},
		A: []byte{0}, AStride: 2,
	}
	if err := validateImage(img); !errors.Is(err, errInvalidImage) {
		t.Fatalf("accepted an NYCbCrA whose alpha plane is too short: %v", err)
	}
}

// The checks must not start rejecting well-formed images, which is the failure
// mode a tightened bound invites.
func TestValidateImageAcceptsWellFormedRasters(t *testing.T) {
	nycbcra := image.NewNYCbCrA(image.Rect(0, 0, 16, 9), image.YCbCrSubsampleRatio420)
	cases := []struct {
		name string
		img  image.Image
	}{
		{"NRGBA", image.NewNRGBA(image.Rect(0, 0, 16, 9))},
		{"RGBA", image.NewRGBA(image.Rect(0, 0, 16, 9))},
		{"Gray", image.NewGray(image.Rect(0, 0, 16, 9))},
		{"YCbCr 420", image.NewYCbCr(image.Rect(0, 0, 16, 9), image.YCbCrSubsampleRatio420)},
		{"YCbCr 444", image.NewYCbCr(image.Rect(0, 0, 16, 9), image.YCbCrSubsampleRatio444)},
		{"NYCbCrA 420", nycbcra},
		{"offset NRGBA", image.NewNRGBA(image.Rect(7, 5, 23, 14))},
		{"offset YCbCr 422", image.NewYCbCr(image.Rect(7, 5, 23, 14), image.YCbCrSubsampleRatio422)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateImage(tc.img); err != nil {
				t.Fatalf("rejected a well-formed image: %v", err)
			}
		})
	}
}
