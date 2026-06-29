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
