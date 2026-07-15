//go:build jabcode_gpu

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
