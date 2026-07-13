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

// Diagnose runs the authoritative decoder once with detailed observation and
// renders that trace as text and annotated images. Diagnostics never replay a
// route, add a decode attempt or influence which route wins.
func Diagnose(img image.Image, w io.Writer, imageDir, sourceName string) ([]byte, error) {
	return DiagnoseProfile(img, w, imageDir, sourceName, wire.CReference)
}

// DiagnoseProfile is Diagnose under the selected wire-format profile.
func DiagnoseProfile(img image.Image, w io.Writer, imageDir, sourceName string, profile wire.Profile) ([]byte, error) {
	sink := newDiagImageSink(imageDir, w, sourceName)
	data, trace, err := read.DecodeWithTraceProfile(img, profile)
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
func logFinderPass(w io.Writer, label string, p detect.FinderPassStats) {
	diagLogf(w, "pass %s:", label)
	diagLogf(w, "  rawHits (n-1-1-1-m, horiz+conditional vert) = %d", p.RawHits)
	diagLogf(w, "  branch routing: blue(->FP0/FP3)=%d  red(->FP1/FP2)=%d", p.BranchBlue, p.BranchRed)
	diagLogf(w, "  red path: colorOK(fp2found)=%d  classified(fp1/fp2)=%d", p.RedColor, p.RedClassified)
	diagLogf(w, "  crossCheckPattern survivors  = FP0=%d FP1=%d FP2=%d FP3=%d",
		p.CrossSurvivors[0], p.CrossSurvivors[1], p.CrossSurvivors[2], p.CrossSurvivors[3])
	diagLogf(w, "  pre-prune groups (fc>=3)     = FP0=%d FP1=%d FP2=%d FP3=%d",
		p.Preprune[0], p.Preprune[1], p.Preprune[2], p.Preprune[3])
	diagLogf(w, "  selected foundCount (post-prune) = FP0=%d FP1=%d FP2=%d FP3=%d",
		p.Selected[0], p.Selected[1], p.Selected[2], p.Selected[3])
	diagLogf(w, "  missing=%d  status=%s  interpolated=%v", p.Missing, statusName(p.Status), p.Interpolated)
	for _, c := range p.Candidates {
		diagLogf(w, "    cand typ=%d center=(%.0f,%.0f) foundCount=%d moduleSize=%.1f", c.Typ, c.Center.X, c.Center.Y, c.FoundCount, c.ModuleSize)
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
func diagPalette(w io.Writer, pal []byte, colorNumber int, profile wire.Profile) {
	copies := spec.PaletteCopies(colorNumber)
	canonical := palette.SetDefaultProfile(colorNumber, profile)
	if copies <= 0 || canonical == nil || len(pal) < colorNumber*3*copies {
		diagLogf(w, "  palette dump skipped (colorNumber=%d len=%d)", colorNumber, len(pal))
		return
	}
	names4 := []string{"blk", "mag", "yel", "cyn"}
	if profile == wire.ISO23634 {
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
