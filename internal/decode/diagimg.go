package decode

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// diagImageSink writes the per-stage annotated images of one Diagnose run into
// a directory, so a failure is visible at a glance rather than only in report
// numbers. A nil sink disables all rendering and every method is nil-safe, so
// call sites stay unconditional. Filenames are numbered in emission order and
// carry the sink's context prefix (set while replaying an orientation rung or
// region crop), so the files sort in the same order as the report's stages.
// Like the rest of the diagnostic, it observes and never influences decoding.
type diagImageSink struct {
	dir    string
	w      io.Writer // the report writer, for the "wrote ..." lines
	seq    *int      // shared across prefix contexts so numbering stays global
	prefix string
}

// newDiagImageSink returns a sink writing into dir, creating it if needed, or
// nil (rendering disabled) when dir is empty or cannot be created. Stage
// images from a previous run would interleave with this run's numbering, so
// files matching the sink's own naming scheme (a digit-run prefix before an
// underscore, .png suffix) are removed first.
func newDiagImageSink(dir string, w io.Writer) *diagImageSink {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		diagLogf(w, "diag images: %v (images disabled)", err)
		return nil
	}
	if ents, err := os.ReadDir(dir); err == nil {
		for _, e := range ents {
			n := e.Name()
			i := 0
			for i < len(n) && n[i] >= '0' && n[i] <= '9' {
				i++
			}
			if i > 0 && i < len(n) && n[i] == '_' && strings.HasSuffix(n, ".png") {
				os.Remove(filepath.Join(dir, n))
			}
		}
	}
	return &diagImageSink{dir: dir, w: w, seq: new(int)}
}

// withPrefix returns a sink whose stage names carry the parent's prefix plus
// the given segment, so nested contexts (a rung inside a region) compose. The
// emission counter is shared with the parent so filenames keep sorting in
// report order.
func (s *diagImageSink) withPrefix(prefix string) *diagImageSink {
	if s == nil {
		return nil
	}
	return &diagImageSink{dir: s.dir, w: s.w, seq: s.seq, prefix: s.prefix + prefix}
}

// save writes img as the next numbered stage PNG.
func (s *diagImageSink) save(name string, img image.Image) {
	if s == nil {
		return
	}
	*s.seq++
	// Three digits keep lexical order intact for runs beyond 99 stages (a
	// multi-region, multi-rung failure replay can emit that many).
	path := filepath.Join(s.dir, fmt.Sprintf("%03d_%s%s.png", *s.seq, s.prefix, name))
	f, err := os.Create(path)
	if err != nil {
		diagLogf(s.w, "diag images: %v", err)
		return
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		diagLogf(s.w, "diag images: encode %s: %v", path, err)
		return
	}
	diagLogf(s.w, "diag image written: %s", path)
}

// Overlay colours are picked off the eight module hues (which are the RGB cube
// corners), so annotations stay tellable from symbol content.
var (
	diagColROI  = color.NRGBA{255, 128, 0, 255} // orange
	diagColQuad = color.NRGBA{0, 255, 128, 255} // spring green
	diagColGrid = color.NRGBA{255, 128, 0, 255} // orange
	diagColType = [4]color.NRGBA{               // finder candidates by type
		{255, 128, 0, 255}, // FP0 orange
		{0, 255, 128, 255}, // FP1 spring green
		{255, 0, 128, 255}, // FP2 pink
		{128, 0, 255, 255}, // FP3 violet
	}
)

// diagOverlayBase clones img into a zero-based mutable canvas for annotation.
func diagOverlayBase(img image.Image) *image.NRGBA {
	b := img.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	return dst
}

// diagBitmapImage converts a detector bitmap back into an image for annotation.
func diagBitmapImage(bm *Bitmap) *image.NRGBA {
	dst := image.NewNRGBA(image.Rect(0, 0, bm.Width, bm.Height))
	for y := range bm.Height {
		for x := range bm.Width {
			o := bm.Offset(x, y)
			var c color.NRGBA
			if bm.Channels >= 3 {
				c = color.NRGBA{bm.Pix[o], bm.Pix[o+1], bm.Pix[o+2], 255}
			} else {
				v := bm.Pix[o]
				c = color.NRGBA{v, v, v, 255}
			}
			dst.SetNRGBA(x, y, c)
		}
	}
	return dst
}

// diagStroke derives the overlay line thickness from the canvas size, so
// annotations stay visible on high-resolution captures.
func diagStroke(b image.Rectangle) int {
	return max(2, max(b.Dx(), b.Dy())/500)
}

// diagFill sets a filled th-sided square centred at (x, y), clipped to dst.
func diagFill(dst *image.NRGBA, x, y, th int, c color.NRGBA) {
	b := dst.Bounds()
	for dy := -th / 2; dy <= th/2; dy++ {
		for dx := -th / 2; dx <= th/2; dx++ {
			px, py := x+dx, y+dy
			if px >= b.Min.X && px < b.Max.X && py >= b.Min.Y && py < b.Max.Y {
				dst.SetNRGBA(px, py, c)
			}
		}
	}
}

// diagLine draws a straight segment from a to b.
func diagLine(dst *image.NRGBA, a, b PointF, th int, c color.NRGBA) {
	steps := int(math.Hypot(b.X-a.X, b.Y-a.Y)) + 1
	for i := 0; i <= steps; i++ {
		t := float64(i) / float64(steps)
		diagFill(dst, int(a.X+t*(b.X-a.X)), int(a.Y+t*(b.Y-a.Y)), th, c)
	}
}

// diagRect draws the outline of r.
func diagRect(dst *image.NRGBA, r image.Rectangle, th int, c color.NRGBA) {
	tl := PointF{float64(r.Min.X), float64(r.Min.Y)}
	tr := PointF{float64(r.Max.X - 1), float64(r.Min.Y)}
	br := PointF{float64(r.Max.X - 1), float64(r.Max.Y - 1)}
	bl := PointF{float64(r.Min.X), float64(r.Max.Y - 1)}
	diagLine(dst, tl, tr, th, c)
	diagLine(dst, tr, br, th, c)
	diagLine(dst, br, bl, th, c)
	diagLine(dst, bl, tl, th, c)
}

// diagCrossMark draws a diagonal cross centred at p - diagonal so it stays
// tellable from the axis-aligned grid and box overlays.
func diagCrossMark(dst *image.NRGBA, p PointF, arm, th int, c color.NRGBA) {
	a := float64(arm)
	diagLine(dst, PointF{p.X - a, p.Y - a}, PointF{p.X + a, p.Y + a}, th, c)
	diagLine(dst, PointF{p.X - a, p.Y + a}, PointF{p.X + a, p.Y - a}, th, c)
}

// saveBinarized writes the three binarized channel masks as one composite: each
// mask lands in its output channel, so every pixel shows its 3-bit colour
// classification directly - the composite reads as a posterized version of the
// capture when binarization is healthy, and washes out where it is not.
func (s *diagImageSink) saveBinarized(name string, ch [3]*Bitmap) {
	if s == nil || ch[0] == nil || ch[1] == nil || ch[2] == nil {
		return
	}
	w, h := ch[0].Width, ch[0].Height
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			o := ch[0].Offset(x, y)
			dst.SetNRGBA(x, y, color.NRGBA{ch[0].Pix[o], ch[1].Pix[o], ch[2].Pix[o], 255})
		}
	}
	s.save(name, dst)
}

// saveMatrixClassified writes each sampled module as the canonical colour of
// its classified palette index - the classifier's view of the symbol, using
// the same per-corner palettes, normalization and black thresholds the decoder
// uses. Held against the raw sampled matrix, classification flips pop out.
func (s *diagImageSink) saveMatrixClassified(name string, matrix *Bitmap, symbol *DecodedSymbol) {
	if s == nil || matrix == nil || symbol == nil || len(symbol.Palette) == 0 {
		return
	}
	colorNumber := len(symbol.Palette) / 3 / spec.ColorPaletteNumber
	canon := palette.SetDefault(colorNumber)
	if canon == nil {
		return
	}
	normPalette := make([]float64, colorNumber*4*spec.ColorPaletteNumber)
	NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range spec.ColorPaletteNumber {
		t := GetPaletteThreshold(symbol.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}
	scale := min(32, max(4, 1024/max(matrix.Width, matrix.Height)))
	dst := image.NewNRGBA(image.Rect(0, 0, matrix.Width*scale, matrix.Height*scale))
	for y := range matrix.Height {
		for x := range matrix.Width {
			idx := int(DecodeModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, x, y))
			var c color.NRGBA
			if idx*3+2 < len(canon) {
				c = color.NRGBA{canon[idx*3], canon[idx*3+1], canon[idx*3+2], 255}
			}
			for dy := range scale {
				for dx := range scale {
					dst.SetNRGBA(x*scale+dx, y*scale+dy, c)
				}
			}
		}
	}
	s.save(name, dst)
}

// saveROIs writes the input with each proposed region's box, in rank order.
func (s *diagImageSink) saveROIs(img image.Image, rois []ROICandidate) {
	if s == nil || len(rois) == 0 {
		return
	}
	dst := diagOverlayBase(img)
	th := diagStroke(dst.Bounds())
	org := img.Bounds().Min
	// Reverse rank order, so the top-ranked box draws last (on top) and thicker.
	for i := len(rois) - 1; i >= 0; i-- {
		extra := 0
		if i == 0 {
			extra = th
		}
		diagRect(dst, rois[i].Bounds.Sub(org), th+extra, diagColROI)
	}
	s.save("rois", dst)
}

// saveFinders writes the frame with every finder candidate of the pass marked
// by a cross coloured per type, plus the selected quad's edges when one exists.
func (s *diagImageSink) saveFinders(bm *Bitmap, cands []FinderPattern, quad []FinderPattern) {
	if s == nil {
		return
	}
	dst := diagBitmapImage(bm)
	th := diagStroke(dst.Bounds())
	for _, c := range cands {
		arm := max(3*th, int(c.ModuleSize*2.5))
		diagCrossMark(dst, c.Center, arm, th, diagColType[c.Typ&3])
	}
	if len(quad) == 4 {
		// Perimeter in placement order: FP0 top-left, FP1 top-right,
		// FP2 bottom-right, FP3 bottom-left.
		for i := range 4 {
			diagLine(dst, quad[i].Center, quad[(i+1)%4].Center, th, diagColQuad)
		}
	}
	s.save("finders", dst)
}

// saveGrid writes the frame with the module grid warped through the same
// transform sampling uses; a misaligned grid shows a bad quad or side size
// immediately.
func (s *diagImageSink) saveGrid(bm *Bitmap, pt Perspective, side image.Point) {
	if s == nil {
		return
	}
	dst := diagBitmapImage(bm)
	th := max(1, diagStroke(dst.Bounds())/2)
	// Perspective bends module rows/columns, so each grid line is drawn as a
	// polyline through every module boundary rather than one segment.
	for x := 0; x <= side.X; x++ {
		prev := pt.Warp(PointF{float64(x), 0})
		for y := 1; y <= side.Y; y++ {
			p := pt.Warp(PointF{float64(x), float64(y)})
			diagLine(dst, prev, p, th, diagColGrid)
			prev = p
		}
	}
	for y := 0; y <= side.Y; y++ {
		prev := pt.Warp(PointF{0, float64(y)})
		for x := 1; x <= side.X; x++ {
			p := pt.Warp(PointF{float64(x), float64(y)})
			diagLine(dst, prev, p, th, diagColGrid)
			prev = p
		}
	}
	s.save("grid", dst)
}

// saveMatrix writes the sampled module matrix upscaled with hard module edges,
// so palette damage and misclassification patterns are visible directly.
func (s *diagImageSink) saveMatrix(name string, matrix *Bitmap) {
	if s == nil || matrix == nil {
		return
	}
	scale := min(32, max(4, 1024/max(matrix.Width, matrix.Height)))
	dst := image.NewNRGBA(image.Rect(0, 0, matrix.Width*scale, matrix.Height*scale))
	for y := range matrix.Height {
		for x := range matrix.Width {
			o := matrix.Offset(x, y)
			c := color.NRGBA{matrix.Pix[o], matrix.Pix[o+1], matrix.Pix[o+2], 255}
			for dy := range scale {
				for dx := range scale {
					dst.SetNRGBA(x*scale+dx, y*scale+dy, c)
				}
			}
		}
	}
	s.save(name, dst)
}

// savePalette writes the palettes read from the symbol as swatch rows - the
// canonical palette on top, then one row per embedded corner palette - so a
// cast or a misaligned palette walk shows up as off-colour swatches.
func (s *diagImageSink) savePalette(name string, symbol *DecodedSymbol) {
	if s == nil || symbol == nil || len(symbol.Palette) == 0 {
		return
	}
	colorNumber := len(symbol.Palette) / 3 / spec.ColorPaletteNumber
	canon := palette.SetDefault(colorNumber)
	if canon == nil {
		return
	}
	const cell, gap = 48, 4
	rows := 1 + spec.ColorPaletteNumber
	dst := image.NewNRGBA(image.Rect(0, 0, colorNumber*(cell+gap)+gap, rows*(cell+gap)+gap))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.NRGBA{160, 160, 160, 255}), image.Point{}, draw.Src)
	fill := func(row, col int, c color.NRGBA) {
		x0 := gap + col*(cell+gap)
		y0 := gap + row*(cell+gap)
		draw.Draw(dst, image.Rect(x0, y0, x0+cell, y0+cell), image.NewUniform(c), image.Point{}, draw.Src)
	}
	for i := range colorNumber {
		fill(0, i, color.NRGBA{canon[i*3], canon[i*3+1], canon[i*3+2], 255})
	}
	for p := range spec.ColorPaletteNumber {
		for i := range colorNumber {
			o := (p*colorNumber + i) * 3
			fill(1+p, i, color.NRGBA{symbol.Palette[o], symbol.Palette[o+1], symbol.Palette[o+2], 255})
		}
	}
	s.save(name, dst)
}
