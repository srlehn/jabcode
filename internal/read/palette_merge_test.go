package read

import (
	"bytes"
	"image"
	"image/draw"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
)

// boxBlur is an exact box blur with edge clamping, kept local so this gate runs
// in the default build rather than only under the degradation harness.
func boxBlur(src image.Image, r int) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	in := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(in, in.Bounds(), src, b.Min, draw.Src)
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	n := float64((2*r + 1) * (2*r + 1))
	for y := range h {
		for x := range w {
			var sr, sg, sb float64
			for dy := -r; dy <= r; dy++ {
				for dx := -r; dx <= r; dx++ {
					i := in.PixOffset(min(max(x+dx, 0), w-1), min(max(y+dy, 0), h-1))
					sr += float64(in.Pix[i+0])
					sg += float64(in.Pix[i+1])
					sb += float64(in.Pix[i+2])
				}
			}
			o := out.PixOffset(x, y)
			out.Pix[o+0] = byte(sr/n + 0.5)
			out.Pix[o+1] = byte(sg/n + 0.5)
			out.Pix[o+2] = byte(sb/n + 0.5)
			out.Pix[o+3] = 255
		}
	}
	return out
}

// A large symbol under a blur wide enough to touch a whole module is where the
// embedded palette read breaks down: every palette entry comes from a single
// module, so once neighbouring colour bleeds into it each of the four corner
// copies is wrong on its own, and classifying against the nearest copy alone
// then misreads enough data modules to put the payload past LDPC. The sampled
// grid is not the problem - it carries the correct colour on every module here -
// which is why this failed while every geometry measure said the symbol was
// found exactly.
//
// The sizes are the ones that actually break: the largest version at 6 pixels
// per module, and the rectangular 32x8, both under radius 2. Smaller versions
// and gentler blurs decode from the per-corner palettes alone and would pass
// with the merged-palette retry removed.
func TestBlurredLargeSymbolsDecode(t *testing.T) {
	payload := []byte("version detection gate: large and rectangular symbols under mild degradation 0123456789")
	for _, version := range []image.Point{{X: 32, Y: 32}, {X: 32, Y: 8}} {
		r, err := encode.Render(encode.Config{
			Colors: 8, ModuleSize: 6, SymbolNumber: 1,
			SymbolVersions: []image.Point{version},
		}, payload)
		if err != nil {
			t.Fatalf("version=%v: %v", version, err)
		}
		got, err := Decode(boxBlur(r.Image, 2))
		if err != nil {
			t.Errorf("version=%v: %v", version, err)
			continue
		}
		if want := append([]byte("]j1"), payload...); !bytes.Equal(got, want) {
			t.Errorf("version=%v: payload = %q, want %q", version, got, want)
		}
	}
}

// Stream reads a single frame through its own correction path rather than
// through detectPrimary's, so a payload-correction step added for Decode alone
// leaves the two disagreeing on exactly the frames that need it. A stream
// handed one undamaged frame must reach the same verdict as Decode does.
func TestBlurredLargeSymbolDecodesThroughStream(t *testing.T) {
	payload := []byte("version detection gate: large and rectangular symbols under mild degradation 0123456789")
	r, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 6, SymbolNumber: 1,
		SymbolVersions: []image.Point{{X: 32, Y: 32}},
	}, payload)
	if err != nil {
		t.Fatal(err)
	}
	frame := boxBlur(r.Image, 2)

	var s Stream
	got, err := s.Decode(frame)
	if err != nil {
		t.Fatalf("stream did not decode a frame Decode reads: %v", err)
	}
	if want := append([]byte("]j1"), payload...); !bytes.Equal(got, want) {
		t.Fatalf("stream payload = %q, want %q", got, want)
	}
}
