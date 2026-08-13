//go:build !js

package detect

import "testing"

func TestGPUMetadataRetainedBytesCoversEveryParityRowSet(t *testing.T) {
	want := (gpuMetadataParamWords+gpuMetadataRecordWords+
		gpuMetadataLDPCRowSets*gpuMetadataLDPCRowWords)*4 + gpuMetadataStaticBytes
	if gpuMetadataRetainedBytes != want {
		t.Fatalf("metadata retained bytes = %d, want %d", gpuMetadataRetainedBytes, want)
	}
}
