package core

import "image"

// Bitmap is a raw 8-bit-per-channel pixel buffer used by the detector and
// decoder. Channels is 4 (RGBA) for input images, or 1 for grayscale/binary
// intermediates.
type Bitmap struct {
	// Unlike image.NRGBA/image.Gray, the buffer is always tightly packed
	// with a zero origin (no Stride, no sub-image offsets); flat pixel
	// loops rely on that.

	Width, Height int
	Channels      int
	Pix           []byte // row-major, Width*Height*Channels bytes
}

func NewBitmap(width, height, channels int) *Bitmap {
	return &Bitmap{Width: width, Height: height, Channels: channels, Pix: make([]byte, width*height*channels)}
}

// BitmapFromImage converts any image.Image into a 4-channel RGBA bitmap. Reading
// a JAB Code from a file is therefore just stdlib decoding (e.g. png.Decode)
// followed by this conversion.
func BitmapFromImage(img image.Image) *Bitmap {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	bm := NewBitmap(w, h, 4)
	i := 0
	for y := range h {
		for x := range w {
			r, g, bl, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			bm.Pix[i+0] = byte(r >> 8)
			bm.Pix[i+1] = byte(g >> 8)
			bm.Pix[i+2] = byte(bl >> 8)
			bm.Pix[i+3] = byte(a >> 8)
			i += 4
		}
	}
	return bm
}

// Offset returns the index into Pix of pixel (x, y).
func (b *Bitmap) Offset(x, y int) int { return (y*b.Width + x) * b.Channels }
