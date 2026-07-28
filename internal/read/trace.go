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
// ladder rung (kind), which pyramid level (-1 on the single-level path, -2 on
// the enlarged scale), on which proposed region (-1
// for the whole frame). side carries the finder-based locate estimate once the
// locate got that far - the grid the finder-pattern sample used, which a
// wrong-geometry failure needs recorded. It is NOT necessarily the grid of the
// last decode attempt: the alignment-pattern fallback resamples at the
// metadata-derived version size without updating the attempt.
type routeAttempt struct {
	kind  string
	level int
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
	// levels is the pyramid depth searched, 1 on the single-scale path. Slot
	// traces leave it zero: only the root trace the read was started with sees
	// the whole search.
	levels int

	detailed      bool
	pyramid       []image.Point
	pyramidImages []image.Image
	rois          []DiagnosticROIs
	details       []DiagnosticAttempt
}

// setLevels records the pyramid depth the read searched.
func (tr *routeTrace) setLevels(n int) {
	if tr == nil {
		return
	}
	tr.levels = n
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
	tr.rois = append(tr.rois, other.rois...)
	tr.details = append(tr.details, other.details...)
}

// beginAttempt opens a detailed attempt record; the route's kind is named once,
// at the matching finishAttempt, and stamped onto this record there.
func (tr *routeTrace) beginAttempt(roi int) *DiagnosticAttempt {
	if tr == nil || !tr.detailed {
		return nil
	}
	return &DiagnosticAttempt{Route: DiagnosticRoute{Level: tr.level, ROI: roi}}
}

func (tr *routeTrace) finishAttempt(a routeAttempt, detail *DiagnosticAttempt, payload []byte) {
	tr.add(a)
	if tr == nil || detail == nil {
		return
	}
	detail.Route.Kind = a.kind
	detail.Stage = a.stage.String()
	detail.Side = a.side
	detail.Payload = append([]byte(nil), payload...)
	tr.details = append(tr.details, *detail)
}

// winner returns the route whose read produced the returned payload, and ok
// false for a failed read. Every ladder merges its slot traces in commit order
// and returns at the slot that decoded, so the first decoded attempt in the
// collected order is the one that won; no separate bookkeeping is needed to
// name it.
func (tr *routeTrace) winner() (routeAttempt, bool) {
	if tr == nil {
		return routeAttempt{}, false
	}
	for _, a := range tr.attempts {
		if a.stage == readDecoded {
			return a, true
		}
	}
	return routeAttempt{}, false
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
