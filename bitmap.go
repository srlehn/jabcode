package jabcode

import "image"

// bitmap is a raw 8-bit-per-channel pixel buffer used by the detector and
// decoder, mirroring the reference jab_bitmap. channels is 4 (RGBA) for input
// images, or 1 for grayscale/binary intermediates.
type bitmap struct {
	width, height int
	channels      int
	pix           []byte // row-major, width*height*channels bytes
}

func newBitmap(width, height, channels int) *bitmap {
	return &bitmap{width: width, height: height, channels: channels, pix: make([]byte, width*height*channels)}
}

// bitmapFromImage converts any image.Image into a 4-channel RGBA bitmap. Reading
// a JAB Code from a file is therefore just stdlib decoding (e.g. png.Decode)
// followed by this conversion.
func bitmapFromImage(img image.Image) *bitmap {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	bm := newBitmap(w, h, 4)
	i := 0
	for y := range h {
		for x := range w {
			r, g, bl, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			bm.pix[i+0] = byte(r >> 8)
			bm.pix[i+1] = byte(g >> 8)
			bm.pix[i+2] = byte(bl >> 8)
			bm.pix[i+3] = byte(a >> 8)
			i += 4
		}
	}
	return bm
}

// at returns a pointer offset into pix for pixel (x, y).
func (b *bitmap) offset(x, y int) int { return (y*b.width + x) * b.channels }
