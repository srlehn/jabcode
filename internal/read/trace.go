package read

import "image"

// readStage identifies how far one read attempt got before it stopped, in
// pipeline order: finder location, side-size estimation, symbol sampling,
// decoding. readAborted marks an attempt cancelled between stages (a pyramid
// route told to quit), which never happens on a fully failed read - the
// attribution case - because no route ever wins there.
type readStage int

const (
	readAborted readStage = iota
	readNoFinders
	readNoSideSize
	readNoSample
	readSampled
	readDecoded
)

func (s readStage) String() string {
	switch s {
	case readAborted:
		return "aborted"
	case readNoFinders:
		return "no-finders"
	case readNoSideSize:
		return "no-side-size"
	case readNoSample:
		return "no-sample"
	case readSampled:
		return "sampled"
	case readDecoded:
		return "decoded"
	default:
		return "unknown"
	}
}

// routeAttempt records one attempted read route and how far it got: which
// pyramid level (-1 on the single-level path), under which pre-rotation, on
// which proposed region (-1 for the whole frame). side carries the
// finder-based locate estimate once the locate got that far - the grid the
// finder-pattern sample used, which a wrong-geometry failure needs recorded.
// It is NOT necessarily the grid of the last decode attempt: the
// alignment-pattern fallback resamples at the metadata-derived version size
// without updating the attempt.
type routeAttempt struct {
	level int
	deg   float64
	roi   int
	stage readStage
	side  image.Point
}

// routeTrace collects the attempts of one full read so a diagnostic consumer
// (the capture harness) can attribute a failure to the furthest stage an
// attempted route reached, instead of guessing from an upright-only view.
// It is observation-only: no decode decision reads it, and every method is
// nil-safe so the production path threads nil at zero cost. The pyramid
// gives each route slot its own trace and merges them in slot order after
// the join, so the collected order is deterministic; a successful read may
// return before all slots are joined and then carries a partial trace (its
// purpose is failure attribution, where every route runs to completion).
// Seeded cross-level resampling has its own route record, so its sampling and
// decode progress is attributed directly rather than to the locating route.
type routeTrace struct {
	// level stamps attempts added directly to this trace; the pyramid sets it
	// per slot, the single-level path uses -1.
	level    int
	attempts []routeAttempt

	detailed bool
	pyramid  []image.Point
	probes   []DiagnosticProbe
	rois     []DiagnosticROIs
	details  []DiagnosticAttempt
}

// add records one attempt, stamping the trace's level.
func (tr *routeTrace) add(a routeAttempt) {
	if tr == nil {
		return
	}
	a.level = tr.level
	tr.attempts = append(tr.attempts, a)
}

// merge appends another trace's attempts verbatim (they keep their own level
// stamps).
func (tr *routeTrace) merge(other *routeTrace) {
	if tr == nil || other == nil {
		return
	}
	tr.attempts = append(tr.attempts, other.attempts...)
	tr.probes = append(tr.probes, other.probes...)
	tr.rois = append(tr.rois, other.rois...)
	tr.details = append(tr.details, other.details...)
}

func (tr *routeTrace) beginAttempt(kind string, deg float64, roi int) *DiagnosticAttempt {
	if tr == nil || !tr.detailed {
		return nil
	}
	return &DiagnosticAttempt{Route: DiagnosticRoute{Kind: kind, Level: tr.level, Angle: deg, ROI: roi}}
}

func (tr *routeTrace) finishAttempt(a routeAttempt, detail *DiagnosticAttempt, payload []byte) {
	tr.add(a)
	if tr == nil || detail == nil {
		return
	}
	detail.Stage = a.stage.String()
	detail.Side = a.side
	detail.Payload = append([]byte(nil), payload...)
	tr.details = append(tr.details, *detail)
}

// best returns the attempt that reached the furthest stage; ties keep the
// earliest attempt, so the deterministic route order breaks them. ok is false
// when nothing was attempted.
func (tr *routeTrace) best() (best routeAttempt, ok bool) {
	if tr == nil {
		return routeAttempt{}, false
	}
	for _, a := range tr.attempts {
		if !ok || a.stage > best.stage {
			best, ok = a, true
		}
	}
	return best, ok
}
