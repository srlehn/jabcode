package diag

import (
	"fmt"
	"io"
	"strings"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/read"
)

func renderTrace(w io.Writer, sink *diagImageSink, trace *read.DiagnosticTrace) {
	if trace == nil {
		diagLogf(w, "diagnostic trace unavailable")
		return
	}
	sink.save("input", trace.Input)
	if len(trace.Pyramid) > 0 {
		diagLogf(w, "pyramid levels: %v", trace.Pyramid)
	}
	for i, level := range trace.PyramidImages {
		sink.withPrefix(fmt.Sprintf("pyramid_level%02d_", i)).save("input", level)
	}
	for i := range trace.Probes {
		renderProbeTrace(w, sink.withPrefix(fmt.Sprintf("probe%02d_", i+1)), trace.Probes[i])
	}
	for i := range trace.ROIs {
		renderROITrace(w, sink.withPrefix(fmt.Sprintf("roi_set%02d_", i+1)), trace.ROIs[i])
	}
	for i := range trace.Attempts {
		a := &trace.Attempts[i]
		prefix := fmt.Sprintf("attempt%02d_%s_", i+1, routeToken(a.Route))
		renderAttemptTrace(w, sink.withPrefix(prefix), i+1, a)
	}
}

func routeToken(route read.DiagnosticRoute) string {
	level := "full"
	if route.Level >= 0 {
		level = fmt.Sprintf("level%02d", route.Level)
	}
	angle := strings.ReplaceAll(fmt.Sprintf("%03.0f", route.Angle), "-", "m")
	if route.ROI >= 0 {
		return fmt.Sprintf("%s_roi%02d_angle%s", level, route.ROI, angle)
	}
	return fmt.Sprintf("%s_%s_angle%s", level, route.Kind, angle)
}

func renderProbeTrace(w io.Writer, sink *diagImageSink, probe read.DiagnosticProbe) {
	diagLogf(w, "orientation probe [%s] level=%d roi=%d retained=%v", probe.Label, probe.Level, probe.ROI, probe.Rungs)
	sink.save("input", probe.Probe.Input)
	for i, angle := range probe.Probe.Angles {
		diagLogf(w, "  angle %.0f: types=%d survivors=%d", angle.Family.Deg, angle.Family.Types, angle.Family.Sum)
		s := sink.withPrefix(fmt.Sprintf("angle%02d_%03.0f_", i+1, angle.Family.Deg))
		s.save("balanced", diagBitmapImage(angle.Bitmap))
		s.saveBinarized("binarized", angle.Channels)
		s.saveFinders(angle.Bitmap, angle.Pass.Candidates, nil)
	}
}

func renderROITrace(w io.Writer, sink *diagImageSink, rois read.DiagnosticROIs) {
	diagLogf(w, "ROI proposals level=%d: %d", rois.Level, len(rois.Candidates))
	for i, r := range rois.Candidates {
		diagLogf(w, "  ROI %d score=%.3f chromaVar=%.3f grad=%.3f tiles=%d box=(%d,%d)-(%d,%d)",
			i, r.Score, r.ChromaVar, r.GradEnergy, r.Tiles,
			r.Bounds.Min.X, r.Bounds.Min.Y, r.Bounds.Max.X, r.Bounds.Max.Y)
	}
	sink.saveROITileMaps(rois.TileMap)
	sink.saveROIs(rois.Image, rois.Candidates)
}

func renderAttemptTrace(w io.Writer, sink *diagImageSink, index int, attempt *read.DiagnosticAttempt) {
	diagLogf(w, "attempt %d: kind=%s level=%d angle=%.0f roi=%d stage=%s side=(%d,%d)",
		index, attempt.Route.Kind, attempt.Route.Level, attempt.Route.Angle,
		attempt.Route.ROI, attempt.Stage, attempt.Side.X, attempt.Side.Y)
	if attempt.Balanced != nil {
		sink.save("balanced", diagBitmapImage(attempt.Balanced))
	}
	for i, pass := range attempt.Detector.Passes {
		logFinderPass(w, fmt.Sprintf("attempt %d pass %d %s", index, i+1, pass.Label), pass)
		s := sink.withPrefix(fmt.Sprintf("pass%02d_", i+1))
		if i < len(attempt.DetectorTrace.PassInputs) {
			s.save("input", diagBitmapImage(attempt.DetectorTrace.PassInputs[i]))
		}
		if i < len(attempt.DetectorTrace.PassChannels) {
			s.saveBinarized("binarized", attempt.DetectorTrace.PassChannels[i])
		}
		var selected []detect.FinderPattern
		if pass.Status == core.Success {
			selected = attempt.Finders
		}
		s.saveFinders(attempt.Balanced, pass.Candidates, selected)
	}
	if len(attempt.Detector.Passes) == 0 {
		sink.saveBinarized("binarized", attempt.InitialChannels)
	}
	if attempt.HasTransform {
		sink.saveGrid(attempt.Balanced, attempt.Transform, attempt.Side)
	}
	if attempt.PrintDetected && attempt.HasTransform {
		d := attempt.ChannelOffsets
		diagLogf(w, "  channel offsets: R=(%.2f,%.2f) G=(%.2f,%.2f) B=(%.2f,%.2f)",
			d[0].X, d[0].Y, d[1].X, d[1].Y, d[2].X, d[2].Y)
		sink.saveChannelOffsets(attempt.Balanced, attempt.Transform, attempt.Side, d)
	}
	if attempt.Sampled != nil {
		sink.saveMatrix("sampled", attempt.Sampled)
	}
	if attempt.Alignment != nil {
		diagLogf(w, "  alignment: attempted=%v grid=(%d,%d) patterns=%d rectangles=%d reason=%q",
			attempt.Alignment.Attempted, attempt.Alignment.Grid.X, attempt.Alignment.Grid.Y,
			len(attempt.Alignment.Patterns), len(attempt.Alignment.Rectangles), attempt.Alignment.Reason)
		sink.saveAlignment(attempt.Balanced, attempt.Alignment)
		if attempt.Alignment.Matrix != nil {
			sink.saveMatrix("sampled_ap", attempt.Alignment.Matrix)
		}
	}
	for i := range attempt.Primary {
		renderPrimaryTrace(w, sink.withPrefix(fmt.Sprintf("primary%02d_", i+1)), i+1, &attempt.Primary[i])
	}
	for i := range attempt.Secondaries {
		secondary := &attempt.Secondaries[i]
		diagLogf(w, "  secondary %d: host=%d dock=%d result=%s", i+1,
			secondary.HostIndex, secondary.DockedPosition, statusName(secondary.Result))
		s := sink.withPrefix(fmt.Sprintf("secondary%02d_", i+1))
		if secondary.HasTransform {
			s.saveFinders(attempt.Balanced, secondary.Patterns, secondary.Patterns)
			s.saveGrid(attempt.Balanced, secondary.Transform, secondary.Side)
		}
		s.saveMatrix("sampled", secondary.Matrix)
		s.savePalette("palette", &secondary.Symbol)
		s.saveMatrixClassified("classified", secondary.Matrix, &secondary.Symbol, &secondary.Classification)
		s.saveMatrixComparison("sampled_vs_classified", secondary.Matrix, &secondary.Symbol, &secondary.Classification)
	}
	if len(attempt.Payload) > 0 {
		diagLogf(w, "  payload: %d bytes %q", len(attempt.Payload), string(attempt.Payload))
	}
}

func renderPrimaryTrace(w io.Writer, sink *diagImageSink, index int, trace *decode.PrimaryTrace) {
	diagLogf(w, "  primary observation %d:", index)
	if trace.PartIAttempted {
		diagLogf(w, "    metadata part I: %s syndrome=%v default=%v",
			metaRetName(trace.PartIResult), trace.PartISyndromeOK, trace.UsedDefault)
	}
	sink.saveModuleWalk("metadata_part_i_modules", trace.Matrix, nil, trace.PartIDataMap)
	if trace.PaletteAttempted {
		colorNumber := 0
		if trace.Symbol.Meta.NC >= 0 && trace.Symbol.Meta.NC <= 7 {
			colorNumber = 1 << (trace.Symbol.Meta.NC + 1)
		}
		diagLogf(w, "    palette: result=%d colours=%d bytes=%d", trace.PaletteResult,
			colorNumber, len(trace.Symbol.Palette))
	}
	paletteBase := trace.PartIDataMap
	if trace.UsedDefault {
		paletteBase = nil
	}
	sink.saveModuleWalk("palette_modules", trace.Matrix, paletteBase, trace.PaletteDataMap)
	if trace.PartIIAttempted {
		diagLogf(w, "    metadata part II: %s syndrome=%v",
			statusName(trace.PartIIResult), trace.PartIISyndromeOK)
	}
	sink.saveModuleWalk("metadata_part_ii_modules", trace.Matrix, trace.PaletteDataMap, trace.PartIIDataMap)
	if trace.CorrectionAttempted {
		diagLogf(w, "    payload correction: admissionChecked=%v admitted=%v result=%s",
			trace.AdmissionChecked, trace.Admitted, statusName(trace.CorrectionResult))
	} else {
		diagLogf(w, "    payload correction: admissionChecked=%v admitted=%v not attempted",
			trace.AdmissionChecked, trace.Admitted)
	}
	sink.savePalette("palette", &trace.Symbol)
	sink.saveModuleLayout("payload_layout", trace.Matrix, trace.Classification.DataMap)
	sink.saveMatrixClassified("classified", trace.Matrix, &trace.Symbol, &trace.Classification)
	sink.saveMatrixComparison("sampled_vs_classified", trace.Matrix, &trace.Symbol, &trace.Classification)
	if colorNumber, _, ok := diagSymbolPaletteLayout(&trace.Symbol); ok {
		diagPalette(w, trace.Symbol.Palette, colorNumber, trace.Symbol.WireProfile)
	}
}
