package read

import (
	"image"
	"math"
	"slices"

	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/ecc"
)

// evidenceGroup is one bounded provisional content group: retained
// observations of one symbol at one spatial location whose layouts are
// compatible. It is the accumulation scaffolding of the cross-frame step -
// the group holds immutable snapshots in input order; combining their
// evidence is the accumulator's separate job.
//
// Identity discipline: layout agreement (footprint, colour mode, mask, ECC)
// is REJECT-ONLY evidence - those structures are shared across codes, so
// agreement never confirms that two frames carry the same payload, while a
// single mismatch rejects only that observation (capture damage corrupts
// metadata too). Only persistent coherent mismatches reset the group. The
// group is keyed by location: an observation whose quad has moved away from
// the anchor by more than a symbol span belongs to a different track (the
// first iteration enforces single-primary scope by construction - one group
// per Stream).
//
// The ANCHOR is the group's fixed geometric reference and a different object
// from the scheduler's search-prior ring: the ring rotates and decays as the
// camera moves, the anchor stays fixed while the group is active so
// registration never chains frame-to-frame. Snapshots store canonical
// grid-space data, so geometry stays anchor-relative only through track
// identity; one deterministic re-anchor to a strictly better observation is
// allowed, keeping the retained snapshots (their grid data does not depend
// on which frame anchors the track).
type evidenceGroup struct {
	anchor    finding     // frame-coordinate quad of the anchoring observation
	anchorSrc image.Point // source frame size the anchor quad lives in
	anchorFix int         // the anchor's fixed-pattern agreement (re-anchor rank)
	reanchors int         // deterministic re-anchors spent (cap 1)

	side    image.Point                   // layout: sampled grid dimensions
	meta    layoutKey                     // layout: colour mode, mask, ECC, default flag
	snaps   []*decode.ObservationSnapshot // input order, bounded
	rejects int                           // consecutive coherent mismatches toward a reset
	version uint64                        // unique evidence admissions, monotonic within the group

	attempt            evidenceAttempt // last aggregate correction input
	confirmedPayload   []byte          // payload confirmed for the retained canonical signature
	confirmedSignature []byte          // hard bit signature in gross codeword coordinates
	aggregateDisabled  bool            // unsupported layout such as a docked-symbol primary
}

type evidenceAttempt struct {
	version     uint64
	hard        []byte
	syndrome    int
	weightUnits []int
}

// layoutKey is the bit-coordinate compatibility key: evidence may only ever
// combine within one exact layout.
type layoutKey struct {
	nc, mask    int
	ecl         image.Point
	defaultMode bool
}

const (
	evidenceGroupCap   = 8 // snapshots retained per group
	evidenceResetAfter = 3 // consecutive incompatible observations resetting the group

	// Candidate-cost differences are normalized by the captured palette's
	// squared minimum separation. One frame is then capped at one unit of
	// signed evidence per bit, so a single exposure can never dominate a
	// bounded group. The duplicate tolerance lives in those dimensionless
	// units and suppresses observations that add no material new evidence.
	evidenceFrameCap           = 1.0
	evidenceDuplicateTolerance = 1.0 / 64.0
	evidenceStrongBit          = 0.5
	evidenceMinOverlapBits     = 64
	evidenceMaxConflictDivisor = 4 // reject when more than one quarter of strong overlap conflicts
	evidenceConfirmedNumerator = 3 // confirmed reuse requires at least three quarters strong coverage
	evidenceConfirmedDivisor   = 4
)

// evidenceAccumulator is the deterministic reduction of compatible snapshot
// evidence in input order. signed is the additive channel value consumed by
// belief propagation. weight and weightSquared retain enough information for
// the effective sample count (sum(w)^2/sum(w^2)); samples distinguishes no
// opinion from weak observations. frames excludes suppressed duplicates.
type evidenceAccumulator struct {
	signed        []float64
	weight        []float64
	weightSquared []float64
	samples       []int
	frames        int
	duplicates    int
	accepted      [][]float64
}

// calibrateFrameEvidence converts raw squared-distance differences into
// dimensionless bounded channel evidence. Palette separation supplies the
// frame-local distance scale, so uniform brightness scaling changes squared
// costs and the denominator together. Palette-copy disagreement reduces the
// whole frame's authority; module-local masks can refine that weight later.
func calibrateFrameEvidence(raw []float64, separation, disagreement float64) []float64 {
	if len(raw) == 0 || separation <= 0 || disagreement < 0 ||
		math.IsNaN(separation) || math.IsInf(separation, 0) ||
		math.IsNaN(disagreement) || math.IsInf(disagreement, 0) {
		return nil
	}
	scale := separation * separation
	quality := separation / (separation + disagreement)
	out := make([]float64, len(raw))
	for i, v := range raw {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil
		}
		v = v / scale
		v = max(-evidenceFrameCap, min(evidenceFrameCap, v))
		out[i] = v * quality
	}
	return out
}

func snapshotFrameEvidence(s *decode.ObservationSnapshot) []float64 {
	if s == nil || !s.Admitted {
		return nil
	}
	return calibrateFrameEvidence(s.BitEvidence(), s.PaletteSeparation, s.PaletteDisagreement)
}

func nearDuplicateEvidence(frame []float64, accepted [][]float64) bool {
	for _, old := range accepted {
		if len(old) != len(frame) {
			continue
		}
		duplicate := true
		for i, v := range frame {
			if math.Abs(v-old[i]) > evidenceDuplicateTolerance {
				duplicate = false
				break
			}
		}
		if duplicate {
			return true
		}
	}
	return false
}

// add retains one calibrated observation unless a previously accepted frame
// already carries the same evidence within the normalized resolution. The
// comparison and reduction both walk stable slice order.
func (a *evidenceAccumulator) add(frame []float64) bool {
	if len(frame) == 0 || (len(a.signed) != 0 && len(frame) != len(a.signed)) {
		return false
	}
	for _, v := range frame {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > evidenceFrameCap {
			return false
		}
	}
	if nearDuplicateEvidence(frame, a.accepted) {
		a.duplicates++
		return false
	}
	if len(a.signed) == 0 {
		a.signed = make([]float64, len(frame))
		a.weight = make([]float64, len(frame))
		a.weightSquared = make([]float64, len(frame))
		a.samples = make([]int, len(frame))
	}
	kept := append([]float64(nil), frame...)
	a.accepted = append(a.accepted, kept)
	a.frames++
	for i, v := range frame {
		w := math.Abs(v)
		if w == 0 {
			continue
		}
		a.signed[i] += v
		a.weight[i] += w
		a.weightSquared[i] += w * w
		a.samples[i]++
	}
	return true
}

func (a *evidenceAccumulator) effectiveSamples(bit int) float64 {
	if bit < 0 || bit >= len(a.weight) || a.weightSquared[bit] == 0 {
		return 0
	}
	return a.weight[bit] * a.weight[bit] / a.weightSquared[bit]
}

func (a *evidenceAccumulator) hardBits() []byte {
	hard := make([]byte, len(a.signed))
	for i, v := range a.signed {
		if v < 0 {
			hard[i] = 1
		}
	}
	return hard
}

// accumulatedEvidence reduces every usable snapshot currently retained by
// the group. Recomputing from the bounded window avoids order-changing
// subtraction when the oldest snapshot is evicted.
func (g *evidenceGroup) accumulatedEvidence() evidenceAccumulator {
	var a evidenceAccumulator
	for _, s := range g.snaps {
		frame := snapshotFrameEvidence(s)
		a.add(frame)
	}
	return a
}

// contentConflict applies a conservative provisional identity gate. Only bits
// strong in both the retained aggregate and the incoming frame vote; sparse or
// weak overlap is no opinion. Widespread strong sign disagreement is evidence
// of a different code at the same location, never noise to average away.
func contentConflict(a *evidenceAccumulator, frame []float64) bool {
	if len(frame) != len(a.signed) {
		return len(a.signed) != 0
	}
	overlap, conflicts := 0, 0
	for i, v := range frame {
		if math.Abs(v) < evidenceStrongBit || math.Abs(a.signed[i]) < evidenceStrongBit {
			continue
		}
		overlap++
		if (v < 0) != (a.signed[i] < 0) {
			conflicts++
		}
	}
	return overlap >= evidenceMinOverlapBits && conflicts*evidenceMaxConflictDivisor > overlap
}

// correctionDue reports whether unique accumulated evidence changed enough
// to justify another LDPC attempt: the first two-frame aggregate, any hard
// decision change, a lower read-only syndrome weight, or a bit crossing a new
// whole unit of bounded evidence. Duplicate frames never advance version.
func (g *evidenceGroup) correctionDue(a *evidenceAccumulator) bool {
	if g.aggregateDisabled || a.frames < 2 || len(a.signed) == 0 || g.attempt.version == g.version {
		return false
	}
	hard := a.hardBits()
	syndrome, _, valid := ecc.LDPCSyndromeWeight(hard, g.meta.ecl.X, g.meta.ecl.Y)
	if !valid {
		return false
	}
	if g.attempt.version == 0 || !slices.Equal(hard, g.attempt.hard) || syndrome < g.attempt.syndrome {
		return true
	}
	for i, w := range a.weight {
		if int(math.Floor(w)) > g.attempt.weightUnits[i] {
			return true
		}
	}
	return false
}

func (g *evidenceGroup) recordAttempt(a *evidenceAccumulator) {
	hard := a.hardBits()
	syndrome, _, _ := ecc.LDPCSyndromeWeight(hard, g.meta.ecl.X, g.meta.ecl.Y)
	units := make([]int, len(a.weight))
	for i, w := range a.weight {
		units[i] = int(math.Floor(w))
	}
	g.attempt = evidenceAttempt{version: g.version, hard: hard, syndrome: syndrome, weightUnits: units}
}

func (g *evidenceGroup) confirm(payload []byte, a *evidenceAccumulator) {
	g.confirmedPayload = append([]byte(nil), payload...)
	g.confirmedSignature = a.hardBits()
}

// confirmedMatch allows a duplicate or newly useful frame to reuse a payload
// only when at least a bounded amount of its strong evidence agrees perfectly
// with the confirmed canonical signature. Any strong conflict forces a real
// decode or a content transition instead of returning stale data.
func (g *evidenceGroup) confirmedMatch(frame []float64) bool {
	if g.confirmedPayload == nil || len(frame) != len(g.confirmedSignature) {
		return false
	}
	overlap := 0
	for i, v := range frame {
		if math.Abs(v) < evidenceStrongBit {
			continue
		}
		overlap++
		if (v < 0) != (g.confirmedSignature[i] != 0) {
			return false
		}
	}
	return overlap >= evidenceMinOverlapBits &&
		overlap*evidenceConfirmedDivisor >= len(frame)*evidenceConfirmedNumerator
}

// snapshotLayout extracts the compatibility key of a snapshot.
func snapshotLayout(s *decode.ObservationSnapshot) layoutKey {
	return layoutKey{nc: s.Meta.NC, mask: s.Meta.MaskType, ecl: s.Meta.ECL, defaultMode: s.Meta.DefaultMode}
}

// sameTrack reports whether a quad observed on a frame of size src plausibly
// sits on the group's spatial track: every corner within one symbol span of
// the anchor's corner, after scaling between frame sizes. A quad further out
// is a different code (or a jump), never fusion input.
func (g *evidenceGroup) sameTrack(f finding, src image.Point) bool {
	if !f.located || g.anchorSrc.X <= 0 || g.anchorSrc.Y <= 0 || src.X <= 0 || src.Y <= 0 {
		return false
	}
	sx := float64(g.anchorSrc.X) / float64(src.X)
	sy := float64(g.anchorSrc.Y) / float64(src.Y)
	span := 0.0
	for i := range 4 {
		dx := g.anchor.quad[i].X - g.anchor.quad[(i+1)%4].X
		dy := g.anchor.quad[i].Y - g.anchor.quad[(i+1)%4].Y
		span += dx*dx + dy*dy
	}
	span /= 4
	for i := range 4 {
		dx := f.quad[i].X*sx - g.anchor.quad[i].X
		dy := f.quad[i].Y*sy - g.anchor.quad[i].Y
		if dx*dx+dy*dy > span {
			return false
		}
	}
	return true
}

// admit offers a snapshot with its frame-coordinate geometry to the group.
// An empty group anchors on the first admitted observation. A compatible
// snapshot is retained in input order (bounded; past the cap the OLDEST
// falls off - later frames carry the fresher photometrics); an incompatible
// one is rejected and counted, and after evidenceResetAfter consecutive
// coherent rejections the group resets empty (the persistent-evidence rule
// for a content change). Rejection never mutates retained evidence. changed
// is false for a compatible near-duplicate, which neither consumes bounded
// storage nor schedules another correction.
func (g *evidenceGroup) admit(s *decode.ObservationSnapshot, f finding, src image.Point) (admitted, changed bool) {
	if s == nil {
		return false, false
	}
	frame := snapshotFrameEvidence(s)
	if s.Admitted && len(frame) == 0 {
		return false, false
	}
	if len(g.snaps) == 0 {
		if !f.located {
			return false, false
		}
		g.anchor = f
		g.anchor.payload = nil
		g.anchorSrc = src
		g.anchorFix = s.FixedAgree
		g.side = s.Side
		g.meta = snapshotLayout(s)
		g.snaps = append(g.snaps, s)
		g.rejects = 0
		g.version = 1
		return true, true
	}
	if s.Side != g.side || snapshotLayout(s) != g.meta || !g.sameTrack(f, src) {
		if g.rejects++; g.rejects >= evidenceResetAfter {
			*g = evidenceGroup{}
		}
		return false, false
	}
	current := g.accumulatedEvidence()
	if len(frame) > 0 && ((g.confirmedPayload != nil && !g.confirmedMatch(frame)) || contentConflict(&current, frame)) {
		if g.rejects++; g.rejects >= evidenceResetAfter {
			*g = evidenceGroup{}
		}
		return false, false
	}
	if len(frame) > 0 && nearDuplicateEvidence(frame, current.accepted) {
		g.rejects = 0
		return true, false
	}
	g.rejects = 0
	if s.FixedAgree > g.anchorFix && g.reanchors == 0 && f.located {
		// The single deterministic re-anchor: a strictly better observation
		// takes over as the geometric reference; retained grid-space
		// snapshots stay valid.
		g.anchor, g.anchorSrc, g.anchorFix = f, src, s.FixedAgree
		g.anchor.payload = nil
		g.reanchors = 1
	}
	if len(g.snaps) >= evidenceGroupCap {
		g.snaps = append(g.snaps[:0], g.snaps[1:]...)
	}
	g.snaps = append(g.snaps, s)
	g.version++
	return true, true
}

// restart replaces provisional or confirmed content with one independently
// decoded observation. A successful single-frame decode is the strongest
// available identity signal and must not inherit older mixed evidence.
func (g *evidenceGroup) restart(s *decode.ObservationSnapshot, f finding, src image.Point) bool {
	*g = evidenceGroup{}
	admitted, _ := g.admit(s, f, src)
	return admitted
}
