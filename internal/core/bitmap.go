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
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	bm := NewBitmap(w, h, 4)
	switch src := img.(type) {
	case *image.NRGBA:
		// Direct-Pix route: per-pixel At would box a color.Color per call.
		ParallelRows(h, func(lo, hi int) {
			for y := lo; y < hi; y++ {
				row := src.Pix[src.PixOffset(bounds.Min.X, bounds.Min.Y+y):]
				out := bm.Pix[y*w*4 : (y+1)*w*4]
				for x := 0; x < w*4; x += 4 {
					if a := row[x+3]; a == 255 {
						copy(out[x:x+4], row[x:x+4])
					} else {
						// Premultiply exactly as color.NRGBA.RGBA does.
						out[x+0] = nrgbaPremul(row[x+0], a)
						out[x+1] = nrgbaPremul(row[x+1], a)
						out[x+2] = nrgbaPremul(row[x+2], a)
						out[x+3] = a
					}
				}
			}
		})
	case *image.Paletted:
		// Palette colours resolve once into a lookup table; out-of-range
		// indices read as zero instead of panicking.
		var lut [256][4]byte
		for i, c := range src.Palette {
			if i > 255 {
				break
			}
			r, g, b, a := c.RGBA()
			lut[i] = [4]byte{byte(r >> 8), byte(g >> 8), byte(b >> 8), byte(a >> 8)}
		}
		ParallelRows(h, func(lo, hi int) {
			for y := lo; y < hi; y++ {
				row := src.Pix[src.PixOffset(bounds.Min.X, bounds.Min.Y+y):]
				out := bm.Pix[y*w*4 : (y+1)*w*4]
				for x := range w {
					copy(out[x*4:x*4+4], lut[row[x]][:])
				}
			}
		})
	default:
		convert := func(lo, hi int) {
			for y := lo; y < hi; y++ {
				i := y * w * 4
				for x := range w {
					r, g, b, a := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
					bm.Pix[i+0] = byte(r >> 8)
					bm.Pix[i+1] = byte(g >> 8)
					bm.Pix[i+2] = byte(b >> 8)
					bm.Pix[i+3] = byte(a >> 8)
					i += 4
				}
			}
		}
		if concurrentReadSafe(img) {
			ParallelRows(h, convert)
		} else {
			convert(0, h)
		}
	}
	return bm
}

// nrgbaPremul reproduces color.NRGBA.RGBA's alpha premultiplication for one
// channel, truncated to 8 bits the same way the generic conversion is.
func nrgbaPremul(v, a byte) byte {
	return byte((uint32(v) * 257 * uint32(a) / 255) >> 8)
}

// concurrentReadSafe reports whether img is a standard-library raster image,
// whose At reads plain memory and is safe from multiple goroutines. An unknown
// implementation could materialize pixels lazily behind At, so it converts
// sequentially.
func concurrentReadSafe(img image.Image) bool {
	switch img.(type) {
	case *image.NRGBA, *image.RGBA, *image.NRGBA64, *image.RGBA64,
		*image.Gray, *image.Gray16, *image.CMYK, *image.Paletted,
		*image.YCbCr, *image.NYCbCrA, *image.Alpha, *image.Alpha16:
		return true
	}
	return false
}

// Offset returns the index into Pix of pixel (x, y).
func (b *Bitmap) Offset(x, y int) int { return (y*b.Width + x) * b.Channels }
