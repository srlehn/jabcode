package read

import (
	"fmt"
	"image"

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
	// Kind is the ladder rung: upright, roi or seeded. Empty when no route was
	// attempted at all.
	Kind string
	// Level is the pyramid level, -1 for the single-scale search and -2 for
	// the enlarged detection scale.
	Level int
	// ROI is the proposed region index, -1 for a whole-frame route.
	ROI int
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

// RouteKinds counts attempted routes per ladder rung.
type RouteKinds struct {
	Upright, Seeded, ROI int
}

// RouteStages counts attempted routes per furthest stage reached.
type RouteStages struct {
	Aborted, NoFinders, NoSideSize, NoSample, Sampled, Decoded int
}

func (r RouteReport) String() string {
	return fmt.Sprintf("decoded=%t kind=%s level=%d levels=%d roi=%d stage=%s grid=%dx%d attempts=%d by=upright:%d,seeded:%d,roi:%d at=aborted:%d,no-finders:%d,no-side-size:%d,no-sample:%d,sampled:%d,decoded:%d",
		r.Decoded, r.Kind, r.Level, r.Levels, r.ROI, r.Stage, r.Side.X, r.Side.Y,
		r.Attempts, r.Kinds.Upright, r.Kinds.Seeded, r.Kinds.ROI,
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
	r := RouteReport{
		Decoded:  decoded,
		Kind:     a.kind,
		Level:    a.level,
		ROI:      a.roi,
		Stage:    a.stage.String(),
		Side:     a.side,
		Attempts: len(tr.attempts),
		Levels:   tr.levels,
	}
	for _, at := range tr.attempts {
		switch at.kind {
		case "upright":
			r.Kinds.Upright++
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
