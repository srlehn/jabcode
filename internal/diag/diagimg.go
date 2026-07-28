package diag

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
	"slices"
	"strconv"
	"strings"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// diagImageSink writes the per-stage annotated images of one Diagnose run into
// a directory, so a failure is visible at a glance rather than only in report
// numbers. A nil sink disables all rendering and every method is nil-safe, so
// call sites stay unconditional. Filenames are numbered in emission order and
// carry the sink's route and stage context prefix, so the files sort in the
// same order as the report's stages.
// Like the rest of the diagnostic, it observes and never influences decoding.
type diagImageSink struct {
	dir        string
	w          io.Writer // the report writer, for the "wrote ..." lines
	seq        *int      // shared across stage contexts for one source prefix
	filePrefix string
	prefix     string
	record     func(string, image.Image)
	// types selects which stage names are written; nil writes every type. One
	// read of a phone capture emits hundreds of full-frame images across a
	// dozen types, and an investigation almost always wants one of them.
	types map[string]bool
}

// DiagImageTypes are the stage names a caller can select. A name is the part of
// the filename after the sequence number and any stage prefix.
var DiagImageTypes = []string{
	"alignment", "balanced", "binarized", "channel_offsets", "classified",
	"finders", "grid", "input", "metadata_part_i_modules",
	"metadata_part_ii_modules", "metadata_sampled", "palette", "palette_modules",
	"payload_layout", "roi_chroma_map", "roi_gradient_map", "roi_joint_map",
	"rois", "sampled", "sampled_ap", "sampled_vs_classified",
}

// newDiagImageSink returns a sink writing into dir, creating it if needed, or
// nil (rendering disabled) when dir is empty or cannot be created. Existing
// images are preserved; numbering continues independently after the largest
// existing sequence for the source filename prefix.
// types restricts writing to those stage names; empty writes every type. An
// unknown name is reported and ignored rather than silently writing nothing.
func newDiagImageSink(dir string, w io.Writer, sourceName string, types []string) *diagImageSink {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		diagLogf(w, "diag images: %v (images disabled)", err)
		return nil
	}
	filePrefix := diagFilePrefix(sourceName)
	seq := 0
	if ents, err := os.ReadDir(dir); err == nil {
		names := make([]string, 0, len(ents))
		for _, e := range ents {
			names = append(names, e.Name())
		}
		seq = diagImageStartSeq(names, filePrefix)
	}
	var selected map[string]bool
	if len(types) > 0 {
		selected = make(map[string]bool, len(types))
		for _, t := range types {
			if !slices.Contains(DiagImageTypes, t) {
				diagLogf(w, "diag images: unknown type %q, known types: %s", t, strings.Join(DiagImageTypes, ","))
				continue
			}
			selected[t] = true
		}
	}
	return &diagImageSink{dir: dir, w: w, seq: &seq, filePrefix: filePrefix, types: selected}
}

// diagFilePrefix turns an input path into a stable, portable filename prefix.
// The command's stdin marker gets a descriptive prefix; an empty or entirely
// punctuation-only name falls back to "image".
func diagFilePrefix(sourceName string) string {
	if sourceName == "-" {
		return "stdin"
	}
	base := filepath.Base(sourceName)
	if ext := filepath.Ext(base); ext != base {
		base = strings.TrimSuffix(base, ext)
	}
	var b strings.Builder
	separator := false
	for _, r := range base {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			b.WriteRune(r)
			separator = false
			continue
		}
		if b.Len() > 0 && !separator {
			b.WriteByte('_')
			separator = true
		}
	}
	if prefix := strings.Trim(b.String(), "_"); prefix != "" {
		return prefix
	}
	return "image"
}

func diagImageStartSeq(names []string, filePrefix string) int {
	seq := 0
	marker := filePrefix + "_"
	for _, n := range names {
		if !strings.HasPrefix(n, marker) || !strings.HasSuffix(n, ".png") {
			continue
		}
		i := len(marker)
		start := i
		for i < len(n) && n[i] >= '0' && n[i] <= '9' {
			i++
		}
		if i > start && i < len(n) && n[i] == '_' {
			if v, err := strconv.Atoi(n[start:i]); err == nil && v > seq {
				seq = v
			}
		}
	}
	return seq
}

// withPrefix returns a sink whose stage names carry the parent's prefix plus
// the given segment, so nested contexts (a rung inside a region) compose. The
// emission counter is shared with the parent so filenames keep sorting in
// report order.
func (s *diagImageSink) withPrefix(prefix string) *diagImageSink {
	if s == nil {
		return nil
	}
	return &diagImageSink{
		dir:        s.dir,
		w:          s.w,
		seq:        s.seq,
		filePrefix: s.filePrefix,
		prefix:     s.prefix + prefix,
		record:     s.record,
		types:      s.types,
	}
}

// skipStage reports whether name is excluded by the type selection, and is what
// a caller checks *before* building a canvas. Filtering only at the point of
// writing would still draw every full-frame overlay, which is most of the cost
// of a diagnostic run.
//
// It advances the sequence for the stage it skips, because save will not run
// for it, and a selective run's numbering has to keep matching a full one's.
func (s *diagImageSink) skipStage(name string) bool {
	if s == nil {
		return true
	}
	if s.types == nil || s.types[stageType(name)] {
		return false
	}
	*s.seq++
	return true
}

// stageType strips a stage name's repetition index, so selecting "alignment"
// selects "alignment01" and "alignment02" as the flag's help implies.
func stageType(name string) string {
	return strings.TrimRight(name, "0123456789")
}

// save writes img as the next numbered stage PNG.
func (s *diagImageSink) save(name string, img image.Image) {
	if s == nil || img == nil {
		return
	}
	// The sequence advances for a filtered-out image too, so a selective run's
	// filenames still line up with a full one's. Callers that build an expensive
	// canvas should have checked skipStage first; this is the backstop for the
	// ones that pass an image they already had.
	*s.seq++
	if s.types != nil && !s.types[stageType(name)] {
		return
	}
	// Three digits keep lexical order intact for runs beyond 99 stages (a
	// multi-level, multi-route diagnostic can emit that many).
	filename := diagImageFilename(s.filePrefix, *s.seq, s.prefix, name)
	if s.record != nil {
		s.record(filename, img)
		return
	}
	path := filepath.Join(s.dir, filename)
	f, err := os.Create(path)
	if err != nil {
		diagLogf(s.w, "diag images: %v", err)
		return
	}
	defer f.Close()
	enc := png.Encoder{CompressionLevel: png.BestCompression}
	if err := enc.Encode(f, diagDownscale(img)); err != nil {
		diagLogf(s.w, "diag images: encode %s: %v", path, err)
		return
	}
	diagLogf(s.w, "diag image written: %s", path)
}

// diagMaxDim caps a written diagnostic image's long side. A full-resolution
// phone capture costs about sixteen megabytes per stage and a single read emits
// hundreds of stages, which is what makes a diagnostic run unusable rather than
// merely large.
//
// The cap is deliberately high. These images are read to judge whether a mark
// sits on a module or beside it, so the reduction has to stay well clear of the
// module pitch: a 4032-pixel capture halves to 2016, where a ten-pixel module is
// still five pixels across. The reduction is by whole factors for the same
// reason - a fractional resample smears exactly the edges being judged.
const diagMaxDim = 2048

// diagDownscale box-averages img down to diagMaxDim on its long side. Averaging
// rather than dropping samples keeps a one-module feature from vanishing at the
// reductions a capture needs, which is the whole point of looking at these.
func diagDownscale(img image.Image) image.Image {
	b := img.Bounds()
	long := max(b.Dx(), b.Dy())
	if long <= diagMaxDim {
		return img
	}
	n := (long + diagMaxDim - 1) / diagMaxDim
	w, h := (b.Dx()+n-1)/n, (b.Dy()+n-1)/n
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	// Overlays are already NRGBA and are the large ones. Going through the
	// image.Image interface for every source pixel of a dozen-megapixel frame,
	// hundreds of times per read, costs more than the encode it feeds.
	src, _ := img.(*image.NRGBA)
	at := func(x, y int) (uint32, uint32, uint32, uint32) {
		if src != nil {
			o := src.PixOffset(b.Min.X+x, b.Min.Y+y)
			p := src.Pix[o : o+4 : o+4]
			return uint32(p[0]), uint32(p[1]), uint32(p[2]), uint32(p[3])
		}
		cr, cg, cb, ca := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
		return cr >> 8, cg >> 8, cb >> 8, ca >> 8
	}
	for y := range h {
		for x := range w {
			var r, g, bl, a, count uint32
			for sy := y * n; sy < min((y+1)*n, b.Dy()); sy++ {
				for sx := x * n; sx < min((x+1)*n, b.Dx()); sx++ {
					cr, cg, cb, ca := at(sx, sy)
					r, g, bl, a = r+cr, g+cg, bl+cb, a+ca
					count++
				}
			}
			if count == 0 {
				continue
			}
			dst.SetNRGBA(x, y, color.NRGBA{
				uint8(r / count), uint8(g / count), uint8(bl / count), uint8(a / count),
			})
		}
	}
	return dst
}

func diagImageFilename(filePrefix string, seq int, stagePrefix, name string) string {
	return fmt.Sprintf("%s_%03d_%s%s.png", filePrefix, seq, stagePrefix, name)
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
	// diagColScan colours one quad per scan direction. The finder log names the
	// same colour per direction, which is what makes an outline in the image
	// traceable back to the direction and the gates that produced it.
	diagColScan = [6]color.NRGBA{
		{255, 0, 0, 255},   // red
		{255, 255, 0, 255}, // yellow
		{0, 255, 255, 255}, // cyan
		{255, 0, 255, 255}, // magenta
		{0, 128, 255, 255}, // azure
		{128, 255, 0, 255}, // chartreuse
	}
	diagColScanName = [6]string{"red", "yellow", "cyan", "magenta", "azure", "chartreuse"}
	// diagColFinal is reserved for the quad sampling actually used, and is
	// deliberately not in the scan palette. A selection outline is a proposal;
	// only this one produced the grid in the neighbouring image, and reading a
	// proposal as the sampled quad has already caused one wrong diagnosis.
	diagColFinal = color.NRGBA{255, 255, 255, 255} // white
)

// diagOverlayBase clones img into a zero-based mutable canvas for annotation.
func diagOverlayBase(img image.Image) *image.NRGBA {
	b := img.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	return dst
}

// diagBitmapImage converts a detector bitmap back into an image for annotation.
func diagBitmapImage(bm *core.Bitmap) *image.NRGBA {
	if bm == nil {
		return nil
	}
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
func diagLine(dst *image.NRGBA, a, b core.PointF, th int, c color.NRGBA) {
	steps := int(math.Hypot(b.X-a.X, b.Y-a.Y)) + 1
	for i := 0; i <= steps; i++ {
		t := float64(i) / float64(steps)
		diagFill(dst, int(a.X+t*(b.X-a.X)), int(a.Y+t*(b.Y-a.Y)), th, c)
	}
}

// diagRect draws the outline of r.
func diagRect(dst *image.NRGBA, r image.Rectangle, th int, c color.NRGBA) {
	tl := core.Pt(float64(r.Min.X), float64(r.Min.Y))
	tr := core.Pt(float64(r.Max.X-1), float64(r.Min.Y))
	br := core.Pt(float64(r.Max.X-1), float64(r.Max.Y-1))
	bl := core.Pt(float64(r.Min.X), float64(r.Max.Y-1))
	diagLine(dst, tl, tr, th, c)
	diagLine(dst, tr, br, th, c)
	diagLine(dst, br, bl, th, c)
	diagLine(dst, bl, tl, th, c)
}

// diagCrossMark draws a diagonal cross centred at p - diagonal so it stays
// tellable from the axis-aligned grid and box overlays.
func diagCrossMark(dst *image.NRGBA, p core.PointF, arm, th int, c color.NRGBA) {
	a := float64(arm)
	diagLine(dst, core.Pt(p.X-a, p.Y-a), core.Pt(p.X+a, p.Y+a), th, c)
	diagLine(dst, core.Pt(p.X-a, p.Y+a), core.Pt(p.X+a, p.Y-a), th, c)
}

// saveBinarized writes the three binarized channel masks as one composite: each
// mask lands in its output channel, so every pixel shows its 3-bit colour
// classification directly - the composite reads as a posterized version of the
// capture when binarization is healthy, and washes out where it is not.
func (s *diagImageSink) saveBinarized(name string, ch [3]*core.Bitmap) {
	if s == nil || ch[0] == nil || ch[1] == nil || ch[2] == nil {
		return
	}
	if s.skipStage(name) {
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

// saveMatrixClassified writes each module as the canonical colour of its
// palette index. Data-module classifications come from the authoritative
// decode; modules reserved from the payload walk are classified only while
// rendering this diagnostic image.
func (s *diagImageSink) saveMatrixClassified(name string, matrix *core.Bitmap, symbol *core.DecodedSymbol, classification *decode.ModuleClassificationTrace) {
	if s == nil {
		return
	}
	if s.skipStage(name) {
		return
	}
	if dst := diagMatrixClassified(matrix, symbol, classification); dst != nil {
		s.save(name, dst)
	}
}

func diagMatrixClassified(matrix *core.Bitmap, symbol *core.DecodedSymbol, classification *decode.ModuleClassificationTrace) *image.NRGBA {
	if matrix == nil || symbol == nil || len(symbol.Palette) == 0 {
		return nil
	}
	colorNumber, copies, ok := diagSymbolPaletteLayout(symbol)
	if !ok {
		return nil
	}
	hasRecorded := classification != nil &&
		classification.Side == image.Pt(matrix.Width, matrix.Height) &&
		len(classification.Colors) == matrix.Width*matrix.Height
	canon := palette.SetDefault(colorNumber)
	if canon == nil {
		return nil
	}
	normPalette := make([]float64, colorNumber*4*copies)
	decode.NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range copies {
		t := decode.PaletteThreshold(symbol.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}
	scale := diagMatrixScale(matrix)
	dst := image.NewNRGBA(image.Rect(0, 0, matrix.Width*scale, matrix.Height*scale))
	for y := range matrix.Height {
		for x := range matrix.Width {
			idx := 255
			if hasRecorded {
				idx = int(classification.Colors[y*matrix.Width+x])
			}
			if idx == 255 {
				idx = int(decode.DecodeModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, x, y))
			}
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
	return dst
}

func (s *diagImageSink) saveModuleWalk(name string, matrix *core.Bitmap, before, after []byte) {
	if s == nil || matrix == nil || len(after) != matrix.Width*matrix.Height {
		return
	}
	if s.skipStage(name) {
		return
	}
	raw := diagMatrixImage(matrix)
	dst := image.NewNRGBA(raw.Bounds())
	draw.Draw(dst, dst.Bounds(), raw, image.Point{}, draw.Src)
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.NRGBA{0, 0, 0, 190}), image.Point{}, draw.Over)
	scale := diagMatrixScale(matrix)
	seen := false
	for i, used := range after {
		if used == 0 {
			continue
		}
		seen = true
		x, y := i%matrix.Width, i/matrix.Width
		r := image.Rect(x*scale, y*scale, (x+1)*scale, (y+1)*scale)
		draw.Draw(dst, r, raw, r.Min, draw.Src)
		c := color.NRGBA{128, 192, 255, 255}
		if len(before) != len(after) || before[i] == 0 {
			c = diagColROI
		}
		diagRect(dst, r, max(1, scale/5), c)
	}
	if seen {
		s.save(name, dst)
	}
}

func (s *diagImageSink) saveModuleLayout(name string, matrix *core.Bitmap, dataMap []byte) {
	if s == nil || matrix == nil || len(dataMap) != matrix.Width*matrix.Height {
		return
	}
	if s.skipStage(name) {
		return
	}
	dst := diagMatrixImage(matrix)
	scale := diagMatrixScale(matrix)
	for i, reserved := range dataMap {
		if reserved == 0 {
			continue
		}
		x, y := i%matrix.Width, i/matrix.Width
		r := image.Rect(x*scale, y*scale, (x+1)*scale, (y+1)*scale)
		diagRect(dst, r, max(1, scale/5), diagColROI)
	}
	s.save(name, dst)
}

// saveMatrixComparison writes raw samples, canonical classifications and an
// absolute RGB-difference heatmap side by side.
func (s *diagImageSink) saveMatrixComparison(name string, matrix *core.Bitmap, symbol *core.DecodedSymbol, classification *decode.ModuleClassificationTrace) {
	if s == nil {
		return
	}
	if s.skipStage(name) {
		return
	}
	raw := diagMatrixImage(matrix)
	classified := diagMatrixClassified(matrix, symbol, classification)
	if raw == nil || classified == nil || raw.Bounds() != classified.Bounds() {
		return
	}
	w, h := raw.Bounds().Dx(), raw.Bounds().Dy()
	const gap = 8
	dst := image.NewNRGBA(image.Rect(0, 0, w*3+gap*2, h))
	draw.Draw(dst, image.Rect(0, 0, w, h), raw, image.Point{}, draw.Src)
	draw.Draw(dst, image.Rect(w+gap, 0, w*2+gap, h), classified, image.Point{}, draw.Src)
	for y := range h {
		for x := range w {
			a := raw.NRGBAAt(x, y)
			b := classified.NRGBAAt(x, y)
			d := max(absByteDiff(a.R, b.R), max(absByteDiff(a.G, b.G), absByteDiff(a.B, b.B)))
			dst.SetNRGBA(w*2+gap*2+x, y, color.NRGBA{d, 0, 255 - d, 255})
		}
	}
	s.save(name, dst)
}

func absByteDiff(a, b byte) byte {
	if a >= b {
		return a - b
	}
	return b - a
}

// saveROIs writes the input with each proposed region's box, in rank order.
func (s *diagImageSink) saveROIs(img image.Image, rois []detect.ROICandidate) {
	if s == nil || len(rois) == 0 {
		return
	}
	if s.skipStage("rois") {
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

// saveROITileMaps renders the exact normalized chroma, gradient and joint-score
// grids the production proposer used.
func (s *diagImageSink) saveROITileMaps(m detect.ROITileMap) {
	if s == nil || m.GX <= 0 || m.GY <= 0 || len(m.Score) != m.GX*m.GY {
		return
	}
	s.saveROIScoreMap("roi_chroma_map", m, m.Chroma, false)
	s.saveROIScoreMap("roi_gradient_map", m, m.Grad, false)
	s.saveROIScoreMap("roi_joint_map", m, m.Score, true)
}

func (s *diagImageSink) saveROIScoreMap(name string, m detect.ROITileMap, values []float64, thresholds bool) {
	if len(values) != m.GX*m.GY {
		return
	}
	if s.skipStage(name) {
		return
	}
	const cell = 16
	peak := 0.0
	for _, v := range values {
		peak = max(peak, v)
	}
	dst := image.NewNRGBA(image.Rect(0, 0, m.GX*cell, m.GY*cell))
	for ty := range m.GY {
		for tx := range m.GX {
			i := ty*m.GX + tx
			v := 0.0
			if peak > 0 {
				v = values[i] / peak
			}
			level := byte(255 * min(1, max(0, v)))
			c := color.NRGBA{level, level / 2, 255 - level, 255}
			if thresholds && v >= detect.ROIThreshold {
				c = color.NRGBA{255, 128, 0, 255}
			} else if thresholds && v >= detect.ROIAnnexThreshold {
				c = color.NRGBA{255, 220, 0, 255}
			}
			r := image.Rect(tx*cell, ty*cell, (tx+1)*cell, (ty+1)*cell)
			draw.Draw(dst, r, image.NewUniform(c), image.Point{}, draw.Src)
		}
	}
	s.save(name, dst)
}

// saveAlignment overlays expected AP locations, resolved AP locations and the
// AP-grid rectangles selected for block sampling.
func (s *diagImageSink) saveAlignment(name string, bm *core.Bitmap, trace *detect.AlignmentTrace) {
	if s == nil || bm == nil || trace == nil || !trace.Attempted ||
		(len(trace.Expected) == 0 && len(trace.Patterns) == 0 && len(trace.Rectangles) == 0) {
		return
	}
	if s.skipStage(name) {
		return
	}
	dst := diagBitmapImage(bm)
	th := diagStroke(dst.Bounds())
	for _, p := range trace.Expected {
		arm := max(3*th, int(p.ModuleSize*2))
		diagCrossMark(dst, p.Center, arm, th, diagColGrid)
	}
	for _, p := range trace.Patterns {
		if p.FoundCount <= 0 {
			continue
		}
		arm := max(4*th, int(p.ModuleSize*2.5))
		diagCrossMark(dst, p.Center, arm, th+1, diagColQuad)
	}
	if trace.Grid.X > 0 {
		for _, r := range trace.Rectangles {
			tl := r.TopLeft.Y*trace.Grid.X + r.TopLeft.X
			tr := r.TopLeft.Y*trace.Grid.X + r.BottomRight.X
			br := r.BottomRight.Y*trace.Grid.X + r.BottomRight.X
			bl := r.BottomRight.Y*trace.Grid.X + r.TopLeft.X
			if tl < 0 || tr < 0 || br < 0 || bl < 0 ||
				tl >= len(trace.Patterns) || tr >= len(trace.Patterns) ||
				br >= len(trace.Patterns) || bl >= len(trace.Patterns) {
				continue
			}
			q := [4]core.PointF{
				trace.Patterns[tl].Center, trace.Patterns[tr].Center,
				trace.Patterns[br].Center, trace.Patterns[bl].Center,
			}
			for i := range 4 {
				diagLine(dst, q[i], q[(i+1)%4], th, diagColType[2])
			}
		}
	}
	s.save(name, dst)
}

// finderOverlay is one family's finder evidence for a pass: every candidate it
// generated, the quad it published, and the quad each scan direction offered
// with the matching per-direction stats.
type finderOverlay struct {
	cands []detect.FinderPattern
	quad  []detect.FinderPattern
	scans []detect.FinderScanTrace
	stats []detect.FinderFamilyScanStats
	// final is the quad the attempt actually sampled, which is not always one of
	// the selections above: recovery paths can replace it after selection ends.
	final []detect.FinderPattern
}

// saveFinders writes the frame with every finder candidate of the pass marked
// by a cross and a module-size circle coloured per type, plus one outline per
// scan direction that produced a quad.
//
// The circle is drawn at the 3.5 modules a finder core spans, so a measured
// module size that disagrees with the symbol is visible against the pattern
// itself rather than only as a number. That is what distinguishes a candidate
// sitting on background clutter from one on a real finder, and it is why the
// alternatives are drawn at all: a direction whose quad hugs the symbol while
// the published one cuts across it identifies a selection failure directly.
func (s *diagImageSink) saveFinders(bm *core.Bitmap, o finderOverlay) {
	if s == nil {
		return
	}
	if s.skipStage("finders") {
		return
	}
	dst := diagBitmapImage(bm)
	th := diagStroke(dst.Bounds())
	for _, c := range o.cands {
		arm := max(3*th, int(c.ModuleSize*2.5))
		diagCrossMark(dst, c.Center, arm, th, diagColType[c.Typ&3])
		diagCircle(dst, c.Center, c.ModuleSize*finderCoreRadius, th, diagColType[c.Typ&3])
	}
	drawn := false
	for _, sc := range o.scans {
		if len(sc.Quad) != 4 || sc.Scan < 0 || sc.Scan >= len(o.stats) {
			continue
		}
		st := o.stats[sc.Scan]
		col := diagColScan[sc.Scan%len(diagColScan)]
		// The published quad is the one the pass acted on, so it is drawn
		// heavier; every other outline is an alternative that existed and lost.
		w := th
		if st.Published {
			w = th + 2
			drawn = true
		}
		// A corner the outvoted-type prune deleted is boxed where it stood. This
		// is the difference between "that corner was never found" and "it was
		// found and discarded for being rarer than the others", which the
		// selection numbers alone cannot separate. The centre test matters as
		// much as the count: a pruned corner can be replaced by an interpolated
		// estimate somewhere else entirely, and that reads as present.
		for t := range 4 {
			if t < len(sc.Pre) && sc.Pre[t].FoundCount > 0 &&
				(sc.Quad[t].FoundCount == 0 || sc.Quad[t].Center != sc.Pre[t].Center) {
				p := sc.Pre[t]
				diagCircle(dst, p.Center, p.ModuleSize*finderCoreRadius, w, col)
				diagRect(dst, diagSquareAt(p.Center, p.ModuleSize*finderCoreRadius), w, col)
				diagCrossMark(dst, p.Center, max(4*w, int(p.ModuleSize*2.5)), w, diagColType[t])
			}
		}
		diagQuadOutline(dst, sc.Quad, w, col)
	}
	// Without a trace, or where the row walk published without a directional
	// scan, the published quad is all there is.
	if !drawn && len(o.quad) == 4 {
		diagQuadOutline(dst, o.quad, th+2, diagColQuad)
	}
	// Drawn last and heaviest: this is the one the grid and sampled images came
	// from, and every other outline on this frame is a proposal that did not.
	if len(o.final) == 4 {
		diagQuadOutline(dst, o.final, th+3, diagColFinal)
	}
	s.save("finders", dst)
}

// finderCoreRadius is how far a finder core reaches from its centre, in
// modules: three full layers plus the half module the centre sits in.
const finderCoreRadius = 3.5

// diagQuadOutline draws a finder quad's perimeter in placement order - FP0
// top-left, FP1 top-right, FP2 bottom-right, FP3 bottom-left - and rings each
// corner at its own measured scale. The corner cross keeps its type colour, so
// one mark says both which direction found the quad and which finder it took
// that corner to be.
func diagQuadOutline(dst *image.NRGBA, quad []detect.FinderPattern, th int, c color.NRGBA) {
	// A selection with a hole in it has no perimeter worth drawing, but its
	// corners are still evidence: three true corners and one gap is the shape of
	// a direction that was discarded with the symbol under it.
	whole := true
	for i := range 4 {
		if quad[i].FoundCount == 0 {
			whole = false
		}
	}
	if whole {
		for i := range 4 {
			diagLine(dst, quad[i].Center, quad[(i+1)%4].Center, th, c)
		}
	}
	for _, p := range quad {
		if p.FoundCount == 0 {
			continue
		}
		diagCircle(dst, p.Center, p.ModuleSize*finderCoreRadius, th, c)
		diagCrossMark(dst, p.Center, max(4*th, int(p.ModuleSize*2.5)), th, diagColType[p.Typ&3])
	}
}

// diagSquareAt is the axis-aligned box of half-extent r around p, which boxes a
// mark that would otherwise be one more ring among hundreds.
func diagSquareAt(p core.PointF, r float64) image.Rectangle {
	return image.Rect(int(p.X-r), int(p.Y-r), int(p.X+r)+1, int(p.Y+r)+1)
}

// diagCircle draws a circle outline of radius r centred at p. The step keeps
// consecutive samples under a pixel apart so the outline stays closed at any
// radius the module size produces.
func diagCircle(dst *image.NRGBA, p core.PointF, r float64, th int, c color.NRGBA) {
	if r < 1 {
		return
	}
	steps := max(16, int(2*math.Pi*r))
	for i := range steps {
		a := 2 * math.Pi * float64(i) / float64(steps)
		diagFill(dst, int(p.X+r*math.Cos(a)), int(p.Y+r*math.Sin(a)), th, c)
	}
}

// saveGrid writes the frame with the module grid warped through the same
// transform sampling uses; a misaligned grid shows a bad quad or side size
// immediately.
func (s *diagImageSink) saveGrid(bm *core.Bitmap, pt core.Perspective, side image.Point) {
	if s == nil {
		return
	}
	if s.skipStage("grid") {
		return
	}
	dst := diagBitmapImage(bm)
	th := max(1, diagStroke(dst.Bounds())/2)
	// Perspective bends module rows/columns, so each grid line is drawn as a
	// polyline through every module boundary rather than one segment.
	for x := 0; x <= side.X; x++ {
		prev := pt.Warp(core.Pt(float64(x), 0))
		for y := 1; y <= side.Y; y++ {
			p := pt.Warp(core.Pt(float64(x), float64(y)))
			diagLine(dst, prev, p, th, diagColGrid)
			prev = p
		}
	}
	for y := 0; y <= side.Y; y++ {
		prev := pt.Warp(core.Pt(0, float64(y)))
		for x := 1; x <= side.X; x++ {
			p := pt.Warp(core.Pt(float64(x), float64(y)))
			diagLine(dst, prev, p, th, diagColGrid)
			prev = p
		}
	}
	s.save("grid", dst)
}

// saveChannelOffsets overlays the sparse per-channel sample positions used by
// print-aware sampling. Red uses a horizontal mark, green a vertical mark and
// blue a diagonal cross, so coincident zero-offset positions remain visible.
func (s *diagImageSink) saveChannelOffsets(bm *core.Bitmap, pt core.Perspective, side image.Point, delta [3]core.PointF) {
	if s == nil || bm == nil || side.X <= 0 || side.Y <= 0 {
		return
	}
	if s.skipStage("channel_offsets") {
		return
	}
	dst := diagBitmapImage(bm)
	th := max(1, diagStroke(dst.Bounds())/2)
	arm := max(2, th*2)
	stride := max(1, max(side.X, side.Y)/16)
	cols := [3]color.NRGBA{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 96, 255, 255}}
	for y := 0; y < side.Y; y += stride {
		for x := 0; x < side.X; x += stride {
			p := pt.Warp(core.Pt(float64(x)+0.5, float64(y)+0.5))
			for c := range 3 {
				q := core.Pt(p.X+delta[c].X, p.Y+delta[c].Y)
				diagLine(dst, p, q, th, cols[c])
				a := float64(arm)
				switch c {
				case 0:
					diagLine(dst, core.Pt(q.X-a, q.Y), core.Pt(q.X+a, q.Y), th, cols[c])
				case 1:
					diagLine(dst, core.Pt(q.X, q.Y-a), core.Pt(q.X, q.Y+a), th, cols[c])
				case 2:
					diagCrossMark(dst, q, arm, th, cols[c])
				}
			}
		}
	}
	s.save("channel_offsets", dst)
}

// saveMatrix writes the sampled module matrix upscaled with hard module edges,
// so palette damage and misclassification patterns are visible directly.
func (s *diagImageSink) saveMatrix(name string, matrix *core.Bitmap) {
	if s == nil {
		return
	}
	if s.skipStage(name) {
		return
	}
	if dst := diagMatrixImage(matrix); dst != nil {
		s.save(name, dst)
	}
}

func diagMatrixImage(matrix *core.Bitmap) *image.NRGBA {
	if matrix == nil {
		return nil
	}
	scale := diagMatrixScale(matrix)
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
	return dst
}

func diagMatrixScale(matrix *core.Bitmap) int {
	return min(32, max(4, 1024/max(matrix.Width, matrix.Height)))
}

// savePalette writes the palettes read from the symbol as swatch rows - the
// canonical palette on top, then one row per embedded corner palette - so a
// cast or a misaligned palette walk shows up as off-colour swatches.
func (s *diagImageSink) savePalette(name string, symbol *core.DecodedSymbol) {
	if s == nil || symbol == nil || len(symbol.Palette) == 0 {
		return
	}
	if s.skipStage(name) {
		return
	}
	colorNumber, copies, ok := diagSymbolPaletteLayout(symbol)
	if !ok {
		return
	}
	canon := palette.SetDefault(colorNumber)
	if canon == nil {
		return
	}
	const cell, gap = 48, 4
	rows := 1 + copies
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
	for p := range copies {
		for i := range colorNumber {
			o := (p*colorNumber + i) * 3
			fill(1+p, i, color.NRGBA{symbol.Palette[o], symbol.Palette[o+1], symbol.Palette[o+2], 255})
		}
	}
	s.save(name, dst)
}
