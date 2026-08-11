//go:build !js

package detect

import (
	"bytes"
	"os/exec"
	"testing"

	"github.com/srlehn/vulki/shader"
)

func TestGPUShadersPassSPIRVValidation(t *testing.T) {
	validator, err := exec.LookPath("spirv-val")
	if err != nil {
		t.Skip("spirv-val is not installed")
	}
	shaders := []struct {
		name   string
		source string
	}{
		{name: "binarize_rgb", source: binarizeRGBWGSL},
		{name: "filter_binary", source: filterBinaryWGSL},
		{name: "pack_binary_masks", source: packBinaryMasksWGSL},
		{name: "halve_nrgba", source: halveNRGBAWGSL},
		{name: "histogram_rgb", source: histogramRGBWGSL},
		{name: "histogram_bounds", source: histogramBoundsWGSL},
		{name: "balance_rgb", source: balanceRGBWGSL},
		{name: "block_thresholds", source: blockThresholdsWGSL},
		{name: "finder_average", source: finderAverageWGSL},
		{name: "finder_average_reduce", source: finderAverageReduceWGSL},
		{name: "finder_candidates", source: finderCandidatesWGSL},
		{name: "finder_decision", source: finderDecisionWGSL},
		{name: "finder_geometry", source: finderGeometryWGSL},
		{name: "finder_chain_current", source: finderChainBindingsWGSL +
			finderChainPreludeWGSL + finderChainRowWGSL + finderChainCurrentWGSL},
		{name: "sample_symbol", source: sampleSymbolWGSL},
		{name: "ldpc_hard", source: ldpcHardWGSL},
		{name: "ldpc_soft", source: ldpcSoftWGSL},
		{name: "ldpc_soft_graph", source: ldpcSoftGraphWGSL},
		{name: "ldpc_soft_prepare", source: ldpcSoftPrepareWGSL},
		{name: "ldpc_matrix_regular", source: "const MATRIX_SLOT: u32 = 0u;\n" + ldpcMatrixWGSL},
		{name: "ldpc_matrix_tail", source: "const MATRIX_SLOT: u32 = 1u;\n" + ldpcMatrixWGSL},
		{name: "metadata_part1", source: metadataPart1WGSL},
		{name: "metadata_palette", source: metadataPaletteWGSL},
		{name: "metadata_finish", source: metadataFinishWGSL},
		{name: "metadata_payload", source: metadataPayloadWGSL},
		{name: "payload_map", source: payloadMapWGSL},
		{name: "payload_permute", source: payloadPermuteWGSL},
		{name: "primary_result", source: primaryResultWGSL},
		{name: "payload_bits", source: payloadBitsWGSL},
		{name: "payload_reliability", source: payloadReliabilityWGSL},
		{name: "admission_fixed", source: admissionFixedWGSL},
		{name: "local_module_count", source: localModuleCountWGSL},
		{name: "resident_local_module_count", source: residentLocalModuleCountWGSL},
		{name: "pitch_samples", source: pitchSamplesWGSL},
		{name: "pitch_line_sums", source: pitchLineSumsWGSL},
		{name: "pitch_center", source: pitchCenterWGSL},
		{name: "pitch_acf", source: pitchACFWGSL},
		{name: "descreen_horizontal", source: descreenHorizontalWGSL},
		{name: "descreen_vertical", source: descreenVerticalWGSL},
	}
	if finderChainBSIWGSL != "" {
		shaders = append(shaders, struct {
			name   string
			source string
		}{
			name: "finder_chain_bsi",
			source: finderChainBindingsWGSL + finderChainPreludeWGSL +
				finderChainRowWGSL + finderChainBSIWGSL,
		})
	}
	for _, shaderSource := range shaders {
		t.Run(shaderSource.name, func(t *testing.T) {
			spirv, err := shader.Compile(shaderSource.source)
			if err != nil {
				t.Fatalf("compile WGSL: %v", err)
			}
			cmd := exec.Command(validator, "--target-env", "vulkan1.1", "-")
			cmd.Stdin = bytes.NewReader(spirv)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("spirv-val: %v\n%s", err, output)
			}
		})
	}
}
