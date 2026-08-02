package diag

import (
	"fmt"
	"image"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/read"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

// DiagnoseCapabilities runs the authoritative decoder once with detailed
// observation and renders that trace as text and annotated images. Diagnostics
// never replay a route, add a decode attempt or influence which route wins.
// imageTypes restricts which image stages are written; empty writes all of
// DiagImageTypes.
//
// The capability mask is additive, so a diagnosis can be narrowed to one wire
// format or to any subset. Pass the compiled set for an unrestricted run.
func DiagnoseCapabilities(img image.Image, w io.Writer, imageDir, sourceName string, capabilities wire.Capabilities, imageTypes []string) ([]byte, error) {
	sink := newDiagImageSink(imageDir, w, sourceName, imageTypes)
	data, trace, err := read.DecodeWithTraceCapabilities(img, capabilities)
	renderTrace(w, sink, trace)
	if err != nil {
		diagLogf(w, "Decode: FAILED: %v", err)
		return nil, err
	}
	diagLogf(w, "Decode: OK (%d bytes): %q", len(data), string(data))
	return data, nil
}

// diagLogf writes one newline-terminated report line to w.
func diagLogf(w io.Writer, format string, args ...any) {
	fmt.Fprintf(w, format+"\n", args...)
}

// logFinderPass prints one finder-detection pass's counters.
func logFinderPass(w io.Writer, label string, p detect.FinderPassStats, families detect.FinderFamilySet) {
	diagLogf(w, "pass %s:", label)
	if families.Has(detect.FinderFamilyCurrent) {
		logFinderFamilyPass(w, "current ISO/current-C", p.FinderFamilyPassStats, true)
	}
	if bsi, ok := p.BSIFamilyStats(); ok {
		logFinderFamilyPass(w, "BSI/pre-v2.0 C", bsi, false)
	}
}

func logFinderFamilyPass(w io.Writer, label string, p detect.FinderFamilyPassStats, routed bool) {
	diagLogf(w, "  %s signature:", label)
	diagLogf(w, "    rawHits (n-1-1-1-m, horiz+conditional vert) = %d", p.RawHits)
	if routed {
		diagLogf(w, "    branch routing: blue(->FP0/FP3)=%d  red(->FP1/FP2)=%d", p.BranchBlue, p.BranchRed)
		diagLogf(w, "    red path: colorOK(fp2found)=%d  classified(fp1/fp2)=%d", p.RedColor, p.RedClassified)
	}
	diagLogf(w, "    crossCheckPattern survivors  = FP0=%d FP1=%d FP2=%d FP3=%d (summed over %d scan directions)",
		p.CrossSurvivors[0], p.CrossSurvivors[1], p.CrossSurvivors[2], p.CrossSurvivors[3], len(p.Scans))
	for i, s := range p.Scans {
		mark := " "
		if s.Published {
			mark = "*"
		}
		diagLogf(w, "   %s dir=%-4g overlay=%-7s groups(fc>=3)=%d/%d/%d/%d best=%d/%d/%d/%d selected=%d/%d/%d/%d missing=%d status=%s corner=%s consistent=%v",
			mark, s.Degrees, diagColScanName[i%len(diagColScanName)],
			s.Preprune[0], s.Preprune[1], s.Preprune[2], s.Preprune[3],
			s.Preselect[0], s.Preselect[1], s.Preselect[2], s.Preselect[3],
			s.Selected[0], s.Selected[1], s.Selected[2], s.Selected[3],
			s.Missing, statusName(s.Status), s.Corner, s.Consistent)
	}
	diagLogf(w, "    published: missing=%d  status=%s  corner=%s", p.Missing, statusName(p.Status), p.Corner)
	for _, c := range p.Candidates {
		diagLogf(w, "      cand typ=%d center=(%.0f,%.0f) foundCount=%d moduleSize=%.1f", c.Typ, c.Center.X, c.Center.Y, c.FoundCount, c.ModuleSize)
	}
}

// logFinderRejections prints the cross-check funnel and its retained samples.
// A finder that is generated and then discarded is invisible in the survivor
// counts, so without this a missing corner cannot be told from a rejected one.
func logFinderRejections(w io.Writer, index int, t *detect.DetectorTrace) {
	total := 0
	for _, n := range t.RejectCounts {
		total += n
	}
	if total == 0 {
		return
	}
	diagLogf(w, "attempt %d finder rejections, current-family directional scan only (%d total):", index, total)
	for stage, n := range t.RejectCounts {
		if n > 0 {
			diagLogf(w, "    %-18s = %d", detect.FinderStage(stage), n)
		}
	}
	for _, r := range t.Rejections {
		diagLogf(w, "      %-18s pass=%d typ=%d ch=%+d base=%-4g walk=%-6g start=(%.1f,%.1f) ms=%.2f dcc=%d why=%-11s runs=%v",
			r.Stage, r.Pass+1, r.Typ, r.Channel, r.BaseDeg, r.WalkDeg,
			r.Centre.X, r.Centre.Y, r.Module, r.Confirms, r.Reason, r.Runs)
	}
}

func statusName(s int) string {
	switch s {
	case core.Success:
		return "core.Success"
	case core.Failure:
		return "core.Failure"
	case core.FatalError:
		return "core.FatalError"
	default:
		return fmt.Sprintf("status(%d)", s)
	}
}

func diagSymbolPaletteLayout(symbol *core.DecodedSymbol) (colorNumber, copies int, ok bool) {
	if symbol == nil || len(symbol.Palette) == 0 || symbol.Meta.NC < 0 || symbol.Meta.NC > 7 {
		return 0, 0, false
	}
	colorNumber = 1 << (symbol.Meta.NC + 1)
	copies = spec.PaletteCopies(colorNumber)
	if copies <= 0 || len(symbol.Palette) < colorNumber*3*copies {
		return 0, 0, false
	}
	return colorNumber, copies, true
}

// diagPalette reports the embedded palette copies against the canonical
// palette, including their cross-copy disagreement.
func diagPalette(w io.Writer, pal []byte, colorNumber int, variant wire.Variant) {
	copies := spec.PaletteCopies(colorNumber)
	canonical := palette.SetDefaultVariant(colorNumber, variant)
	if copies <= 0 || canonical == nil || len(pal) < colorNumber*3*copies {
		diagLogf(w, "  palette dump skipped (colorNumber=%d len=%d)", colorNumber, len(pal))
		return
	}
	names4 := []string{"blk", "mag", "yel", "cyn"}
	if variant.UsesISO23634Base() {
		names4 = []string{"blk", "cyn", "mag", "yel"}
	}
	names := map[int][]string{
		4: names4,
		8: {"blk", "blu", "grn", "cyn", "red", "mag", "yel", "wht"},
	}[colorNumber]
	for cp := range copies {
		base := cp * colorNumber * 3
		var sumErr float64
		var off [3]float64
		for c := range colorNumber {
			for ch := range 3 {
				d := float64(pal[base+c*3+ch]) - float64(canonical[c*3+ch])
				sumErr += math.Abs(d)
				off[ch] += d
			}
		}
		n := float64(colorNumber)
		line := fmt.Sprintf("  palette copy %d (meanAbsErr=%.0f, offset r%+.0f g%+.0f b%+.0f)",
			cp, sumErr/(n*3), off[0]/n, off[1]/n, off[2]/n)
		if names != nil {
			var b strings.Builder
			for c := range colorNumber {
				fmt.Fprintf(&b, " %s(%3d,%3d,%3d)", names[c], pal[base+c*3], pal[base+c*3+1], pal[base+c*3+2])
			}
			line += ":" + b.String()
		}
		diagLogf(w, "%s", line)
	}

	spreads := make([]float64, colorNumber)
	var total float64
	for c := range colorNumber {
		for ch := range 3 {
			lo, hi := 255.0, 0.0
			for cp := range copies {
				v := float64(pal[cp*colorNumber*3+c*3+ch])
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
			spreads[c] += hi - lo
			total += hi - lo
		}
	}
	diagLogf(w, "  palette mean cross-copy spread = %.1f (%d copies)", total/(float64(colorNumber)*3), copies)
	if names != nil {
		return
	}
	order := make([]int, colorNumber)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return spreads[order[a]] > spreads[order[b]] })
	for _, c := range order[:min(4, colorNumber)] {
		var b strings.Builder
		for cp := range copies {
			base := cp*colorNumber*3 + c*3
			fmt.Fprintf(&b, " copy%d(%3d,%3d,%3d)", cp, pal[base], pal[base+1], pal[base+2])
		}
		diagLogf(w, "  palette colour %3d canonical(%3d,%3d,%3d) spread %.0f:%s",
			c, canonical[c*3], canonical[c*3+1], canonical[c*3+2], spreads[c]/3, b.String())
	}
}

func metaRetName(r int) string {
	if r == decode.MetadataFailed {
		return "decode.MetadataFailed (-> defaults)"
	}
	return statusName(r)
}
