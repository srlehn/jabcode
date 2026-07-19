package jabcode

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestWasmRootBuildAndDependencies(t *testing.T) {
	env := append(os.Environ(), "CGO_ENABLED=0", "GOOS=js", "GOARCH=wasm")

	build := exec.Command("go", "build", "-o", os.DevNull, ".")
	build.Env = env
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("GOOS=js GOARCH=wasm go build .: %v\n%s", err, output)
	}

	list := exec.Command("go", "list", "-deps", ".")
	list.Env = env
	output, err := list.Output()
	if err != nil {
		t.Fatalf("GOOS=js GOARCH=wasm go list -deps .: %v", err)
	}
	for _, dependency := range strings.Fields(string(output)) {
		if dependency == "github.com/srlehn/vulki" ||
			strings.HasPrefix(dependency, "github.com/srlehn/vulki/") ||
			dependency == "github.com/ebitengine/purego" ||
			strings.HasPrefix(dependency, "github.com/ebitengine/purego/") {
			t.Fatalf("js/wasm dependency graph contains native GPU dependency %q", dependency)
		}
	}
}
