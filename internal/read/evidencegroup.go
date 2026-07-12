package read

import (
	"image"

	"github.com/srlehn/jabcode/internal/decode"
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
)

// snapshotLayout extracts the compatibility key of a snapshot.
func snapshotLayout(s *decode.ObservationSnapshot) layoutKey {
	return layoutKey{nc: s.Meta.NC, mask: s.Meta.MaskType, ecl: s.Meta.ECL, defaultMode: s.Meta.DefaultMode}
}

// sameTrack reports whether a quad observed on a frame of size src plausibly
// sits on the group's spatial track: every corner within one symbol span of
// the anchor's corner, after scaling between frame sizes. A quad further out
// is a different code (or a jump), never fusion input.
func (g *evidenceGroup) sameTrack(f finding, src image.Point) bool {
	if !f.located || g.anchorSrc.X == 0 {
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
// for a content change). Rejection never mutates retained evidence.
func (g *evidenceGroup) admit(s *decode.ObservationSnapshot, f finding, src image.Point) (admitted bool) {
	if len(g.snaps) == 0 {
		if !f.located {
			return false
		}
		g.anchor = f
		g.anchor.payload = nil
		g.anchorSrc = src
		g.anchorFix = s.FixedAgree
		g.side = s.Side
		g.meta = snapshotLayout(s)
		g.snaps = append(g.snaps, s)
		g.rejects = 0
		return true
	}
	if s.Side != g.side || snapshotLayout(s) != g.meta || !g.sameTrack(f, src) {
		if g.rejects++; g.rejects >= evidenceResetAfter {
			*g = evidenceGroup{}
		}
		return false
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
	return true
}
