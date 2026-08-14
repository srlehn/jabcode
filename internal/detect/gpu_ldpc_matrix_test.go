//go:build !js

package detect

import (
	"regexp"
	"strconv"
	"testing"

	"github.com/srlehn/jabcode/internal/ldpccatalog"
)

// TestGPULDPCMatrixConstantsMatchShader pins the numbers the host and the
// catalog duplicate from the shader.
//
// The catalog ones matter most. A drifted bound or packing width does not fail
// loudly: the device reads a well-formed catalog at the wrong offset and emits a
// parity matrix that is merely plausible. Hard LDPC has no payload integrity
// check underneath it, so that decodes into silent corruption rather than into
// an error.
func TestGPULDPCMatrixConstantsMatchShader(t *testing.T) {
	for _, want := range []struct {
		name  string
		value int
	}{
		{"MAX_SUB", gpuLDPCMaxSub},
		{"MAX_SUB", ldpccatalog.MaxCapacity},
		{"MIN_COL_WEIGHT", ldpccatalog.MinColWeight},
		{"MIN_ROW_WEIGHT", ldpccatalog.MinRowWeight},
		{"PIVOT_BITS", 12},
		{"CATALOG_FORMAT", 1},
		{"CATALOG_HEADER_WORDS", 4},
		{"CATALOG_PREFIX_WORDS", 4},
		{"CATALOG_GENERATORS", 2},
	} {
		pattern := regexp.MustCompile(`(?m)^const ` + want.name + `: u32 = (\d+)u;`)
		match := pattern.FindStringSubmatch(ldpcMatrixWGSL)
		if match == nil {
			t.Fatalf("shader declares no %s", want.name)
		}
		found, err := strconv.Atoi(match[1])
		if err != nil {
			t.Fatalf("%s: %v", want.name, err)
		}
		if found != want.value {
			t.Fatalf("%s: shader %d, host %d", want.name, found, want.value)
		}
	}
	pivotNone := regexp.MustCompile(`(?m)^const PIVOT_NONE: u32 = 0x([0-9A-F]+)u;`).
		FindStringSubmatch(ldpcMatrixWGSL)
	if pivotNone == nil {
		t.Fatal("shader declares no PIVOT_NONE")
	}
	if found, err := strconv.ParseInt(pivotNone[1], 16, 32); err != nil {
		t.Fatalf("PIVOT_NONE: %v", err)
	} else if int(found) != ldpccatalog.PivotNone {
		t.Fatalf("PIVOT_NONE: shader %#x, catalog %#x", found, ldpccatalog.PivotNone)
	}
}

// TestGPULDPCMatrixCatalogIsAvailable guards the fail-closed boundary from the
// other side: the build compiles a catalog for the ISO generator, so a rejected
// matrix means a wiring defect rather than a missing artifact.
func TestGPULDPCMatrixCatalogIsAvailable(t *testing.T) {
	if !ldpccatalog.Wellformed(ldpccatalog.GeneratorISO) {
		t.Fatal("no ISO pivot catalog is compiled in")
	}
	combined := ldpccatalog.Combined()
	if len(combined) < len(ldpccatalog.Blob(ldpccatalog.GeneratorISO)) {
		t.Fatalf("combined catalog is %d bytes, smaller than the ISO catalog alone", len(combined))
	}
}
