package encoder_test

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestPublicEncoderDependencyGraph(t *testing.T) {
	command := exec.Command("go", "list", "-deps", ".")
	command.Env = append(os.Environ(), "CGO_ENABLED=0")
	output, err := command.Output()
	if err != nil {
		t.Fatalf("go list -deps .: %v", err)
	}

	for _, dependency := range strings.Fields(string(output)) {
		switch {
		case dependency == "github.com/srlehn/jabcode/internal/read",
			dependency == "github.com/srlehn/jabcode/internal/detect",
			dependency == "github.com/srlehn/vulki",
			strings.HasPrefix(dependency, "github.com/srlehn/vulki/"),
			dependency == "github.com/ebitengine/purego",
			strings.HasPrefix(dependency, "github.com/ebitengine/purego/"):
			t.Fatalf("encoder-only dependency graph contains %q", dependency)
		}
	}
}
