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
		{name: "rotate_nrgba", source: rotateNRGBAWGSL},
		{name: "histogram_rgb", source: histogramRGBWGSL},
		{name: "histogram_bounds", source: histogramBoundsWGSL},
		{name: "balance_rgb", source: balanceRGBWGSL},
		{name: "block_thresholds", source: blockThresholdsWGSL},
		{name: "finder_average", source: finderAverageWGSL},
		{name: "pitch_samples", source: pitchSamplesWGSL},
		{name: "descreen_horizontal", source: descreenHorizontalWGSL},
		{name: "descreen_vertical", source: descreenVerticalWGSL},
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
