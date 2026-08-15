//go:build jabcatalog_exhaustive && !js

package detect

import "testing"

// TestGPULDPCMatrixEveryCatalogKey is the wiring gate in full: every legal key
// of every compiled generator, built on the device and held to the host
// reduction. It is tagged for the same reason the catalog's own exhaustive
// gate is: this is minutes of host elimination and as many device submissions,
// which belongs in a deliberate run rather than in the baseline suite.
func TestGPULDPCMatrixEveryCatalogKey(t *testing.T) {
	checkGPULDPCCatalogKeys(t, gpuLDPCCatalogKeys(1))
}
