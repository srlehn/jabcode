//go:build jabharness

package read

import (
	"bytes"
	"fmt"
	"image"
	"math/rand"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
)

// TestNonDefaultPrimaryMetadataReadback checks that a non-default primary symbol
// reads its Part I (colour count) and Part II (side version, ECC level, mask)
// metadata back across small module sizes and common capture degradations, and
// that metadata corrupted past reading falls to a clean decode failure instead
// of a wrong payload.
//
// The symbol has to be non-default to exercise the metadata read path at all: a
// default-mode symbol - eight colours at the default ECC level - carries no
// primary metadata modules, because the decoder rebuilds its parameters from
// fixed defaults (encode.isDefaultMode). Only a non-default ECC level forces
// real Part I and Part II modules onto the symbol, so this encodes at ECC 5.
//
//	go test -tags jabharness -run TestNonDefaultPrimaryMetadataReadback -v ./internal/read
func TestNonDefaultPrimaryMetadataReadback(t *testing.T) {
	payload := []byte("non-default metadata readback: eight colours, non-default ECC 0123456789")
	const nonDefaultECC = 5
	if nonDefaultECC == spec.DefaultECCLevel || nonDefaultECC == 0 {
		t.Fatalf("ECC %d must be a non-default level to force explicit metadata", nonDefaultECC)
	}

	moduleSizes := []int{6, 7, 8, 10}
	degradations := []struct {
		name  string
		level float64
		apply func(image.Image, float64, *rand.Rand) image.Image
	}{
		{"identity", 0, func(s image.Image, _ float64, _ *rand.Rand) image.Image { return s }},
		{"blur-r", 1, boxBlurDeg},
		{"blur-r", 2, boxBlurDeg},
		{"cast", 1, colorCast},
		{"noise-sd", 8, gaussianNoise},
		{"blur+cast", 1, func(s image.Image, l float64, rng *rand.Rand) image.Image {
			return colorCast(boxBlurDeg(s, l, rng), 1, rng)
		}},
	}
	const seed = 1
	want := isoPayload(payload)

	var report bytes.Buffer
	fmt.Fprintf(&report, "%4s %-12s %-8s %-10s %-10s %s\n",
		"px", "degradation", "decode", "partI", "partII", "syndromes")

	corruptRows := 0
	for _, msz := range moduleSizes {
		r, err := encode.Render(encode.Config{
			Colors: 8, ModuleSize: msz, SymbolNumber: 1, ECCLevel: nonDefaultECC,
		}, payload)
		if err != nil {
			t.Fatalf("encode msz=%d: %v", msz, err)
		}
		for _, d := range degradations {
			rng := rand.New(rand.NewSource(seed))
			img := d.apply(r.Image, d.level, rng)

			out, tr, derr := DecodeWithTrace(img)
			decode := "failed"
			if derr == nil && bytes.Equal(out, want) {
				decode = "ok"
			} else if derr == nil {
				decode = "CORRUPT"
			}

			partI, partII, synI, synII := bestMetadataRead(tr)
			fmt.Fprintf(&report, "%4d %-12s %-8s %-10s %-10s I=%v II=%v\n",
				msz, fmt.Sprintf("%s %g", d.name, d.level), decode, partI, partII, synI, synII)

			if decode == "CORRUPT" {
				corruptRows++
			}
			// An undegraded render must read its explicit metadata; a miss here
			// is a regression in the read path itself, so it fails hard while
			// the degraded rows stay informational.
			if d.name == "identity" && partI != "success" {
				t.Errorf("msz=%d identity: Part I read %q, want success", msz, partI)
			}
		}
	}
	t.Logf("non-default primary metadata across module scales:\n%s", report.String())
	// Zero wrong payloads is the binding invariant: when degradation corrupts
	// the metadata past reading, the decode must fail cleanly rather than return
	// err=nil with the wrong parameters and a wrong payload.
	if corruptRows > 0 {
		t.Errorf("returned a wrong payload on %d rows", corruptRows)
	}
}

// bestMetadataRead reduces a diagnostic trace to the strongest primary-metadata
// read any attempt reached: "success" for an explicit Part I/II read, "default"
// for a fall-to-default, "none" when no attempt reached the stage. synI and
// synII report whether that attempt's hard-decoded parts satisfied their LDPC
// parity checks.
func bestMetadataRead(tr *DiagnosticTrace) (partI, partII string, synI, synII bool) {
	partI, partII = "none", "none"
	rank := map[string]int{"none": 0, "default": 1, "success": 2}
	for i := range tr.Attempts {
		for j := range tr.Attempts[i].Primary {
			p := &tr.Attempts[i].Primary[j]
			if !p.PartIAttempted {
				continue
			}
			state := "default"
			if p.PartIResult == core.Success && !p.UsedDefault {
				state = "success"
			}
			if rank[state] > rank[partI] {
				partI = state
				synI = p.PartISyndromeOK
				if p.PartIIAttempted && p.PartIIResult == core.Success {
					partII, synII = "success", p.PartIISyndromeOK
				} else if p.PartIIAttempted {
					partII, synII = "default", p.PartIISyndromeOK
				}
			}
		}
	}
	return partI, partII, synI, synII
}
