// Package testutil provides shared helpers for the test suites of jabcode's
// internal packages. Its main job is locating the single repository-wide
// testdata/ directory so fixtures (golden vectors, reference images) live in one
// place instead of being copied into each package.
package testutil

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

// CaptureDirEnv names the external high-colour capture corpus used only by
// opt-in harness and experiment tests. Keeping the location explicit prevents
// private, multi-hundred-megabyte captures from becoming repository data.
const CaptureDirEnv = "JABCODE_CAPTURE_DIR"

// repoRoot returns the module root by walking up from this file's own location
// until a go.mod is found. It is independent of the calling package's depth and
// of the test's working directory.
func repoRoot() string {
	_, self, _, _ := runtime.Caller(0)
	dir := filepath.Dir(self)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return dir // filesystem root reached; nothing more to try
		}
		dir = parent
	}
}

// TestdataPath returns the absolute path of name within the repository-wide
// testdata/ directory, so tests in any subpackage share one fixtures tree.
func TestdataPath(name string) string {
	return filepath.Join(repoRoot(), "testdata", name)
}

// CapturePath returns the caller-provided capture corpus or skips the test
// when the private corpus is not installed in the current checkout.
func CapturePath(t *testing.T) string {
	t.Helper()
	dir := os.Getenv(CaptureDirEnv)
	if dir == "" {
		t.Skipf("%s is not set; private capture corpus is unavailable", CaptureDirEnv)
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(repoRoot(), dir)
	}
	return dir
}

// MustAtoi parses s as an int, failing the test on error. It is shared by the
// golden-vector tests that read whitespace-delimited reference fixtures.
func MustAtoi(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("atoi %q: %v", s, err)
	}
	return n
}
