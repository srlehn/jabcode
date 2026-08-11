package core

import "testing"

func cleanPrimaryEvidence() PrimaryEvidence {
	return PrimaryEvidence{
		Available:             true,
		MetadataExplicit:      true,
		MetadataPartIInitial:  2,
		MetadataPartIIInitial: 3,
		PaletteSeparation:     40,
		PaletteDisagreement:   4,
		PayloadInitial:        8,
		PayloadCorrections:    3,
	}
}

func TestPrimaryEvidenceAdmission(t *testing.T) {
	clean := cleanPrimaryEvidence()
	if !clean.Admitted() {
		t.Fatal("clean explicit evidence was rejected")
	}
	bad := clean
	bad.PayloadHardResidual = 1
	if bad.Admitted() {
		t.Fatal("nonzero payload residual was admitted")
	}
	fixed := clean
	fixed.MetadataExplicit = false
	fixed.FixedPatternUsed = true
	fixed.FixedAgreements = 12
	fixed.FixedChecks = 24
	if !fixed.Admitted() {
		t.Fatal("admitted fixed-pattern evidence was rejected")
	}
	fixed.FixedAgreements = 9
	if fixed.Admitted() {
		t.Fatal("weak fixed-pattern evidence was admitted")
	}
	soft := clean
	soft.SoftFallbackUsed = true
	soft.PayloadHardResidual = 4
	soft.PayloadSoftIterations = 6
	if !soft.Admitted() {
		t.Fatal("clean soft fallback evidence was rejected")
	}
}

func TestPrimaryEvidenceDominanceIsStrictAndFailClosed(t *testing.T) {
	base := cleanPrimaryEvidence()
	better := base
	better.PayloadInitial--
	if !better.Dominates(base) || base.Dominates(better) {
		t.Fatal("lower correction residual did not dominate in one direction")
	}
	if base.Dominates(base) {
		t.Fatal("equal evidence broke its own tie")
	}
	conflict := better
	conflict.PaletteDisagreement = base.PaletteDisagreement + 1
	if conflict.Dominates(base) || base.Dominates(conflict) {
		t.Fatal("conflicting evidence produced a winner")
	}
	soft := base
	soft.SoftFallbackUsed = true
	soft.PayloadHardResidual = 2
	soft.PayloadSoftIterations = 2
	if !base.Dominates(soft) || soft.Dominates(base) {
		t.Fatal("otherwise equal hard success did not dominate soft fallback")
	}
	incomparable := base
	incomparable.MetadataExplicit = false
	incomparable.FixedPatternUsed = true
	incomparable.FixedAgreements = 12
	incomparable.FixedChecks = 24
	if base.Dominates(incomparable) || incomparable.Dominates(base) {
		t.Fatal("different admission mechanisms were ranked")
	}
}
