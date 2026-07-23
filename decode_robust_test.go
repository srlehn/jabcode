package jabcode

import (
	"image"
	"image/color"
	"math/rand"
	"testing"
)

// TestDecodeNeverPanics feeds assorted synthetic images to Decode and requires it to
// return cleanly (an error, since none is a real JAB Code) rather than panic. Decode
// must be robust to arbitrary input: the detection heuristics can settle on odd
// geometry, and the downstream sampler must never index out of bounds on it.
func TestDecodeNeverPanics(t *testing.T) {
	sizes := []image.Point{{21, 21}, {64, 64}, {200, 200}, {320, 240}, {512, 384}, {640, 480}}
	rng := rand.New(rand.NewSource(1))
	for _, sz := range sizes {
		for variant := range 4 {
			img := synthImage(sz.X, sz.Y, variant, rng)
			if _, err := Decode(img); err == nil {
				t.Errorf("Decode unexpectedly succeeded on synthetic %dx%d variant %d", sz.X, sz.Y, variant)
			}
		}
	}
}

type boundaryImage struct {
	bounds      image.Rectangle
	panicBounds bool
	panicAt     bool
}

func (img boundaryImage) ColorModel() color.Model { return color.NRGBAModel }
func (img boundaryImage) Bounds() image.Rectangle {
	if img.panicBounds {
		panic("hostile Bounds")
	}
	return img.bounds
}
func (img boundaryImage) At(int, int) color.Color {
	if img.panicAt {
		panic("hostile At")
	}
	return color.Black
}

func TestDecodeRejectsInvalidImageBoundaries(t *testing.T) {
	var typedNil *image.NRGBA
	malformedNRGBA := &image.NRGBA{Pix: make([]byte, 4), Stride: 4, Rect: image.Rect(0, 0, 2, 2)}
	malformedRGBA := &image.RGBA{Pix: make([]byte, 7), Stride: 8, Rect: image.Rect(0, 0, 2, 1)}
	malformedPaletted := &image.Paletted{Pix: []byte{1}, Stride: 1, Rect: image.Rect(0, 0, 1, 1), Palette: color.Palette{color.Black}}
	malformedYCbCr := &image.YCbCr{Y: make([]byte, 3), Cb: []byte{0}, Cr: []byte{0}, YStride: 2, CStride: 1, Rect: image.Rect(0, 0, 2, 2)}
	malformedGray := &image.Gray{Pix: []byte{0}, Stride: 1, Rect: image.Rect(0, 0, 2, 1)}
	base := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	malformedOffset := base.SubImage(image.Rect(3, 3, 4, 4)).(*image.NRGBA)
	malformedOffset.Pix = nil
	cases := []struct {
		name string
		img  image.Image
	}{
		{name: "nil", img: nil},
		{name: "typed nil", img: typedNil},
		{name: "empty", img: image.NewNRGBA(image.Rectangle{})},
		{name: "reversed bounds", img: boundaryImage{bounds: image.Rect(2, 2, 1, 1)}},
		{name: "panic Bounds", img: boundaryImage{panicBounds: true}},
		{name: "panic At", img: boundaryImage{bounds: image.Rect(0, 0, 640, 480), panicAt: true}},
		{name: "malformed NRGBA", img: malformedNRGBA},
		{name: "malformed RGBA", img: malformedRGBA},
		{name: "malformed Paletted index", img: malformedPaletted},
		{name: "malformed YCbCr", img: malformedYCbCr},
		{name: "malformed Gray", img: malformedGray},
		{name: "malformed offset subimage", img: malformedOffset},
		{name: "huge bounds", img: boundaryImage{bounds: image.Rect(0, 0, 1<<20, 1<<20)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := []struct {
				name string
				fn   func() error
			}{
				{name: "Decode", fn: func() error { _, err := Decode(tc.img); return err }},
				{name: "DecodeMessage", fn: func() error { _, err := DecodeMessage(tc.img); return err }},
				{name: "StreamDecode", fn: func() error { _, err := NewStream().Decode(tc.img); return err }},
				{name: "StreamDecodeMessage", fn: func() error { _, err := NewStream().DecodeMessage(tc.img); return err }},
			}
			for _, call := range calls {
				t.Run(call.name, func(t *testing.T) {
					defer func() {
						if recovered := recover(); recovered != nil {
							t.Fatalf("panicked: %v", recovered)
						}
					}()
					if err := call.fn(); err == nil {
						t.Fatal("accepted invalid image")
					}
				})
			}
		})
	}
}

// synthImage builds a deterministic test image: uniform grey, random noise, a
// diagonal gradient, or high-contrast blocks (the last most likely to spawn
// finder-like run lengths and reach the deeper detection paths).
func synthImage(w, h, variant int, rng *rand.Rand) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			var c color.NRGBA
			switch variant {
			case 0:
				c = color.NRGBA{128, 128, 128, 255}
			case 1:
				c = color.NRGBA{byte(rng.Intn(256)), byte(rng.Intn(256)), byte(rng.Intn(256)), 255}
			case 2:
				c = color.NRGBA{byte((x + y) % 256), byte(x % 256), byte(y % 256), 255}
			default:
				if (x/8+y/8)%2 == 0 {
					c = color.NRGBA{0, 0, 0, 255}
				} else {
					c = color.NRGBA{255, 255, 255, 255}
				}
			}
			img.Set(x, y, c)
		}
	}
	return img
}
