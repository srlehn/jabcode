package read

import (
	"fmt"
	"image"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// RouteReport attributes one read to a single rung of the decode ladder: the
// route that produced the payload, or - when nothing decoded - the furthest an
// attempted route got. It is the cheap counterpart of DiagnosticTrace, which
// answers the same question only by retaining every attempt's intermediate
// bitmaps: a route census over a whole capture pool needs the attribution
// without the megabytes.
type RouteReport struct {
	// Decoded reports whether Kind names a winning route or only the best
	// failed one.
	Decoded bool
	// Kind is the ladder rung: frame, roi or seeded. It names where the pixels
	// came from - the whole frame at some pyramid level, a region crop, or a
	// coarse level's finding - and says nothing about scan geometry. Read Deg
	// for that. Empty when no route was
	// attempted at all.
	Kind string
	// Level is the pyramid level, -1 for the single-scale search and -2 for
	// the enlarged detection scale.
	Level int
	// ROI is the proposed region index, -1 for a whole-frame route.
	ROI int
	// Deg is the scan direction that produced the quad of the attempt this
	// report names - the winner when Decoded, otherwise the furthest failed
	// attempt, which may well have located one. It is -1 when that attempt
	// located nothing, which zero cannot stand in for: zero is the row walk and
	// a real answer. Every whole-frame pass sweeps the probe directions when its
	// row walk does not settle, so Kind cannot tell a row-settled read from a
	// turned one and this is what does.
	Deg float64
	// Stage is how far the route got, in pipeline order.
	Stage string
	// Side is the finder-based grid estimate, zero when the route never
	// located.
	Side image.Point
	// Attempts counts the routes collected when the read returned. A failed
	// read runs the ladder out, so this is its full route count; a successful
	// one returns as soon as its winner commits and leaves concurrent losing
	// routes unjoined, so there it counts the routes committed up to the
	// winner rather than every route the process ran.
	Attempts int
	// Kinds splits Attempts by rung, which is what says where a read's route
	// budget went rather than only how large it was.
	Kinds RouteKinds
	// Stages splits Attempts by how far each route got. Stage alone reports the
	// furthest route, which on a failed read says only that *something*
	// sampled; the split says how many routes paid for a sample and how many
	// stopped at the finder walk, which is what distinguishes a read that is
	// expensive because it keeps sampling from one that is expensive because it
	// keeps searching.
	Stages RouteStages
	// Levels is the pyramid depth, 1 for the single-scale search. Level alone
	// does not say how coarse a route ran: level 1 is the finest of two and
	// the second coarsest of four.
	Levels int
}

// ProbeDegrees are the scan directions a whole-frame pass sweeps when its row
// walk does not settle. RouteReport.Deg is always one of them, or -1.
func ProbeDegrees() []float64 { return detect.ProbeDegrees() }

// RouteKinds counts attempted routes per ladder rung.
type RouteKinds struct {
	Frame, Seeded, ROI int
}

// RouteStages counts attempted routes per furthest stage reached.
type RouteStages struct {
	Aborted, NoFinders, NoSideSize, NoSample, Sampled, Decoded int
}

func (r RouteReport) String() string {
	return fmt.Sprintf("decoded=%t kind=%s deg=%g level=%d levels=%d roi=%d stage=%s grid=%dx%d attempts=%d by=frame:%d,seeded:%d,roi:%d at=aborted:%d,no-finders:%d,no-side-size:%d,no-sample:%d,sampled:%d,decoded:%d",
		r.Decoded, r.Kind, r.Deg, r.Level, r.Levels, r.ROI, r.Stage, r.Side.X, r.Side.Y,
		r.Attempts, r.Kinds.Frame, r.Kinds.Seeded, r.Kinds.ROI,
		r.Stages.Aborted, r.Stages.NoFinders, r.Stages.NoSideSize, r.Stages.NoSample,
		r.Stages.Sampled, r.Stages.Decoded)
}

// DecodeWithRouteCapabilities runs the same decode as DecodeCapabilities and
// additionally reports which ladder rung answered. The report is observation
// only; collecting it changes no decode decision.
func DecodeWithRouteCapabilities(img image.Image, capabilities wire.Capabilities) ([]byte, RouteReport, error) {
	if err := validateCapabilities(capabilities); err != nil {
		return nil, RouteReport{}, err
	}
	if err := validateImage(img); err != nil {
		return nil, RouteReport{}, err
	}
	tr := &routeTrace{level: -1}
	message, err := decodeRoutesCapabilities(img, tr, capabilities)
	return messageTransmission(message), tr.report(), err
}

// report renders the trace's winning route, or its best failed one.
func (tr *routeTrace) report() RouteReport {
	a, decoded := tr.winner()
	if !decoded {
		a, _ = tr.best()
	}
	if len(tr.attempts) == 0 {
		// best() hands back a zero attempt, whose zero Deg is the row walk.
		a.deg = -1
	}
	r := RouteReport{
		Decoded:  decoded,
		Kind:     a.kind,
		Level:    a.level,
		ROI:      a.roi,
		Stage:    a.stage.String(),
		Side:     a.side,
		Deg:      a.deg,
		Attempts: len(tr.attempts),
		Levels:   tr.levels,
	}
	for _, at := range tr.attempts {
		switch at.kind {
		case "frame":
			r.Kinds.Frame++
		case "seeded":
			r.Kinds.Seeded++
		case "roi":
			r.Kinds.ROI++
		}
		switch at.stage {
		case readAborted:
			r.Stages.Aborted++
		case readNoFinders:
			r.Stages.NoFinders++
		case readNoSideSize:
			r.Stages.NoSideSize++
		case readNoSample:
			r.Stages.NoSample++
		case readSampled:
			r.Stages.Sampled++
		case readDecoded:
			r.Stages.Decoded++
		}
	}
	return r
}
