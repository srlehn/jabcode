package core

import "math"

// PrimaryEvidence is the compact quantitative record behind one corrected
// primary interpretation. A successful parser is deliberately absent: syntax
// is an admission gate applied after this physical and correction evidence,
// never evidence that can break a tie between disagreeing payloads.
type PrimaryEvidence struct {
	Available        bool
	MetadataExplicit bool
	FixedPatternUsed bool
	SoftFallbackUsed bool

	MetadataPartIInitial      uint32
	MetadataPartIResidual     uint32
	MetadataPartICorrections  uint32
	MetadataPartIIInitial     uint32
	MetadataPartIIResidual    uint32
	MetadataPartIICorrections uint32

	PaletteSeparation   float32
	PaletteDisagreement float32
	FixedAgreements     uint32
	FixedChecks         uint32

	PayloadInitial        uint32
	PayloadHardResidual   uint32
	PayloadCorrections    uint32
	PayloadSoftResidual   uint32
	PayloadSoftIterations uint32
}

// Admitted reports whether the evidence itself permits final message parsing.
// Explicit metadata needs clean parity. A default or weak-metadata route must
// instead carry the fixed-pattern evidence that admitted it. Payload correction
// always ends at zero residual, including a soft retry when one was used.
func (e PrimaryEvidence) Admitted() bool {
	if !e.Available || !finiteNonnegative(e.PaletteSeparation) ||
		!finiteNonnegative(e.PaletteDisagreement) {
		return false
	}
	if e.MetadataExplicit {
		if e.MetadataPartIResidual != 0 || e.MetadataPartIIResidual != 0 {
			return false
		}
	} else if !e.FixedPatternUsed {
		return false
	}
	if e.FixedPatternUsed {
		if e.FixedChecks < 20 || e.FixedAgreements > e.FixedChecks ||
			uint64(e.FixedAgreements)*5 < uint64(e.FixedChecks)*2 {
			return false
		}
	}
	if e.SoftFallbackUsed {
		return e.PayloadSoftResidual == 0 && e.PayloadSoftIterations != 0
	}
	return e.PayloadHardResidual == 0
}

func finiteNonnegative(value float32) bool {
	return value >= 0 && !math.IsInf(float64(value), 0) && !math.IsNaN(float64(value))
}

// Dominates reports whether a is no worse on every comparable signal and
// strictly better on at least one. Incomparable admission mechanisms and any
// conflict return false, so neither compile order nor completion order can
// become an implicit winner rule.
func (a PrimaryEvidence) Dominates(b PrimaryEvidence) bool {
	if !a.Admitted() || !b.Admitted() ||
		a.MetadataExplicit != b.MetadataExplicit ||
		a.FixedPatternUsed != b.FixedPatternUsed {
		return false
	}
	better := false
	lower := func(left, right uint32) bool {
		if left > right {
			return false
		}
		better = better || left < right
		return true
	}
	higherFloat := func(left, right float32) bool {
		if left < right {
			return false
		}
		better = better || left > right
		return true
	}
	lowerFloat := func(left, right float32) bool {
		if left > right {
			return false
		}
		better = better || left < right
		return true
	}
	if !lower(a.MetadataPartIInitial, b.MetadataPartIInitial) ||
		!lower(a.MetadataPartIResidual, b.MetadataPartIResidual) ||
		!lower(a.MetadataPartICorrections, b.MetadataPartICorrections) ||
		!lower(a.MetadataPartIIInitial, b.MetadataPartIIInitial) ||
		!lower(a.MetadataPartIIResidual, b.MetadataPartIIResidual) ||
		!lower(a.MetadataPartIICorrections, b.MetadataPartIICorrections) ||
		!higherFloat(a.PaletteSeparation, b.PaletteSeparation) ||
		!lowerFloat(a.PaletteDisagreement, b.PaletteDisagreement) ||
		!lower(a.PayloadInitial, b.PayloadInitial) ||
		!lower(a.PayloadHardResidual, b.PayloadHardResidual) ||
		!lower(a.PayloadCorrections, b.PayloadCorrections) {
		return false
	}
	if a.FixedPatternUsed {
		left := uint64(a.FixedAgreements) * uint64(b.FixedChecks)
		right := uint64(b.FixedAgreements) * uint64(a.FixedChecks)
		if left < right || a.FixedChecks < b.FixedChecks {
			return false
		}
		better = better || left > right || a.FixedChecks > b.FixedChecks
	}
	if a.SoftFallbackUsed && !b.SoftFallbackUsed {
		return false
	}
	if !a.SoftFallbackUsed && b.SoftFallbackUsed {
		better = true
	} else if a.SoftFallbackUsed {
		if !lower(a.PayloadSoftResidual, b.PayloadSoftResidual) ||
			!lower(a.PayloadSoftIterations, b.PayloadSoftIterations) {
			return false
		}
	}
	return better
}
