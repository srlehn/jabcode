package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
)

// ObservationSnapshot is the deep-owned, immutable form of an observation
// that may outlive its read attempt. A live PrimaryObservation aliases a
// mutable DecodedSymbol (the alignment-pattern retry re-observes into the
// same symbol) and the decoders mutate their buffers in place, so retained
// evidence must be copied out before any further attempt runs: one failed
// correction must never corrupt what a later frame reuses. The snapshot
// carries the layout hypothesis, the interpreted metadata with its syndrome
// status, the captured palette, the sampled module values and the admission
// measurements - everything the cross-frame accumulator consumes. Finder
// geometry in frame coordinates lives with the caller's banked entry (the
// read layer owns coordinates); the snapshot describes the sampled grid.
type ObservationSnapshot struct {
	Side             image.Point   // sampled matrix dimensions, the layout hypothesis
	Meta             core.Metadata // interpreted metadata values (value copy)
	PartISyndromeOK  bool
	PartIISyndromeOK bool
	Palette          []byte // embedded palette as captured, deep copy
	Modules          []byte // sampled module values, matrix pixel layout, deep copy
	Channels         int    // bytes per module in Modules
	DataMap          []byte // complete reserved-module map (metadata, palette, finder, alignment), deep copy

	FixedAgree, FixedChecked               int
	PaletteDisagreement, PaletteSeparation float64
	Admitted                               bool
}

// Snapshot freezes the observation into a deep-owned immutable copy,
// computing the admission measurements once. The receiver stays usable; the
// snapshot shares no memory with it.
func (obs *PrimaryObservation) Snapshot() *ObservationSnapshot {
	agree, checked := obs.FixedPatternAgreement()
	dis, sep := obs.PaletteCoherence()
	dm := append([]byte(nil), obs.dataMap...)
	// Complete the copy with the fixed patterns the payload stage marks, so
	// the snapshot's data-module set matches what a correction would read.
	fillDataMap(dm, obs.Matrix.Width, obs.Matrix.Height, 0)
	s := &ObservationSnapshot{
		Side:             image.Pt(obs.Matrix.Width, obs.Matrix.Height),
		Meta:             obs.Symbol.Meta,
		PartISyndromeOK:  obs.PartISyndromeOK,
		PartIISyndromeOK: obs.PartIISyndromeOK,
		Palette:          append([]byte(nil), obs.Symbol.Palette...),
		Modules:          append([]byte(nil), obs.Matrix.Pix...),
		Channels:         obs.Matrix.Channels,
		DataMap:          dm,

		FixedAgree:          agree,
		FixedChecked:        checked,
		PaletteDisagreement: dis,
		PaletteSeparation:   sep,
		Admitted:            obs.AdmitPayloadCorrection(),
	}
	return s
}
