package jabcode

import (
	"image"
	"math"
	"testing"
)

// benchmarkDecode encodes once and measures repeated decodes of the same image.
func benchmarkDecode(b *testing.B, transform func(image.Image) image.Image, opts ...Option) {
	payload := multiPayload(100)
	img, err := NewEncoder(opts...).Encode(payload)
	if err != nil {
		b.Fatalf("encode: %v", err)
	}
	if transform != nil {
		img = transform(img)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Decode(img); err != nil {
			b.Fatalf("decode: %v", err)
		}
	}
}

func BenchmarkDecodeSingle(b *testing.B) {
	benchmarkDecode(b, nil)
}

func BenchmarkDecodeFourColor(b *testing.B) {
	benchmarkDecode(b, nil, WithColors(4))
}

func BenchmarkDecodeCascade(b *testing.B) {
	v4 := image.Pt(4, 4)
	benchmarkDecode(b, nil, WithSymbols(
		[]int{0, 1, 2, 3, 4, 6, 7, 9, 10},
		[]image.Point{v4, v4, v4, v4, v4, v4, v4, v4, v4},
		[]int{0, 0, 0, 0, 0, 0, 0, 0, 0},
	))
}

// BenchmarkDecodeRotated measures the coarse orientation search plus the
// full-resolution rung decode: the upright pass fails at 30 degrees, so every
// iteration pays the downscaled probe before the counter-rotated read.
func BenchmarkDecodeRotated(b *testing.B) {
	benchmarkDecode(b, func(img image.Image) image.Image {
		return rotateForBench(img, 30)
	})
}

// rotateForBench rotates img about its centre onto a white canvas sized to the
// rotated bounding box, bilinearly interpolated. Test-only: the library's own
// rotation helpers are internal.
func rotateForBench(src image.Image, angleDeg float64) image.Image {
	rad := angleDeg * math.Pi / 180
	sin, cos := math.Sin(rad), math.Cos(rad)
	sb := src.Bounds()
	w, h := sb.Dx(), sb.Dy()
	nw := int(math.Abs(float64(w)*cos) + math.Abs(float64(h)*sin) + 0.5)
	nh := int(math.Abs(float64(w)*sin) + math.Abs(float64(h)*cos) + 0.5)
	dst := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	cxS, cyS := float64(w)/2, float64(h)/2
	cxD, cyD := float64(nw)/2, float64(nh)/2
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			// Inverse-rotate the destination pixel into the source frame.
			dx, dy := float64(x)+0.5-cxD, float64(y)+0.5-cyD
			sx := cos*dx - sin*dy + cxS - 0.5
			sy := sin*dx + cos*dy + cyS - 0.5
			i := dst.PixOffset(x, y)
			x0, y0 := int(math.Floor(sx)), int(math.Floor(sy))
			if x0 < 0 || y0 < 0 || x0+1 >= w || y0+1 >= h {
				dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3] = 255, 255, 255, 255
				continue
			}
			fx, fy := sx-float64(x0), sy-float64(y0)
			for c := 0; c < 3; c++ {
				v := (1-fx)*(1-fy)*channelAt(src, sb.Min.X+x0, sb.Min.Y+y0, c) +
					fx*(1-fy)*channelAt(src, sb.Min.X+x0+1, sb.Min.Y+y0, c) +
					(1-fx)*fy*channelAt(src, sb.Min.X+x0, sb.Min.Y+y0+1, c) +
					fx*fy*channelAt(src, sb.Min.X+x0+1, sb.Min.Y+y0+1, c)
				dst.Pix[i+c] = byte(v + 0.5)
			}
			dst.Pix[i+3] = 255
		}
	}
	return dst
}

func channelAt(img image.Image, x, y, c int) float64 {
	r, g, b, _ := img.At(x, y).RGBA()
	switch c {
	case 0:
		return float64(r >> 8)
	case 1:
		return float64(g >> 8)
	default:
		return float64(b >> 8)
	}
}
