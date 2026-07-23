package core

import (
	"image"
	"image/color"
)

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
	readPixel     func(x, y int) byte
}

// Pixel returns a single-channel pixel without requiring a deferred mask to
// expand into a full image. Ordinary bitmaps take the direct row-major path;
// sparse readers are used only by downstream consumers that need bounded
// windows of a packed detector result.
func (b *Bitmap) Pixel(x, y int) byte {
	if b == nil || x < 0 || y < 0 || x >= b.Width || y >= b.Height {
		return 0
	}
	if b.Pix != nil {
		return b.Pix[(y*b.Width+x)*b.Channels]
	}
	if b.readPixel != nil {
		return b.readPixel(x, y)
	}
	return 0
}

// SetPixelReader installs a deferred reader for a shape-only single-channel
// bitmap. It is an internal handoff for packed detector masks, not a general
// replacement for Bitmap.Pix.
func (b *Bitmap) SetPixelReader(read func(x, y int) byte) {
	b.readPixel = read
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
	case *image.RGBA:
		// Premultiplied bytes round-trip: RGBA() scales each component by
		// 0x101 and the conversion truncates it back to the original byte.
		ParallelRows(h, func(lo, hi int) {
			for y := lo; y < hi; y++ {
				row := src.Pix[src.PixOffset(bounds.Min.X, bounds.Min.Y+y):]
				copy(bm.Pix[y*w*4:(y+1)*w*4], row[:w*4])
			}
		})
	case *image.YCbCr:
		// The struct method call keeps the stdlib conversion math while
		// avoiding the per-pixel color.Color boxing of the At route.
		ParallelRows(h, func(lo, hi int) {
			for y := lo; y < hi; y++ {
				i := y * w * 4
				for x := range w {
					yi := src.YOffset(bounds.Min.X+x, bounds.Min.Y+y)
					ci := src.COffset(bounds.Min.X+x, bounds.Min.Y+y)
					r, g, b, _ := (color.YCbCr{Y: src.Y[yi], Cb: src.Cb[ci], Cr: src.Cr[ci]}).RGBA()
					bm.Pix[i+0] = byte(r >> 8)
					bm.Pix[i+1] = byte(g >> 8)
					bm.Pix[i+2] = byte(b >> 8)
					bm.Pix[i+3] = 255
					i += 4
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

// NRGBA returns a zero-copy image view of a 4-channel bitmap, valid because the
// buffer is always tightly packed with a zero origin. The view aliases Pix, so
// writes through either side are visible in both. Returns nil for other channel
// counts or for a shape-only deferred bitmap.
func (b *Bitmap) NRGBA() *image.NRGBA {
	if b == nil || b.Channels != 4 || b.Pix == nil {
		return nil
	}
	return &image.NRGBA{Pix: b.Pix, Stride: b.Width * 4, Rect: image.Rect(0, 0, b.Width, b.Height)}
}

// Gray returns a zero-copy image view of a materialized single-channel bitmap.
// Bitmap owns the backing bytes and the returned image aliases them; callers
// must not retain the view after the bitmap's pixels are replaced.
func (b *Bitmap) Gray() *image.Gray {
	if b == nil || b.Channels != 1 || b.Pix == nil {
		return nil
	}
	return &image.Gray{Pix: b.Pix, Stride: b.Width, Rect: image.Rect(0, 0, b.Width, b.Height)}
}
