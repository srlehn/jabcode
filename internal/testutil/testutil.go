// Package testutil provides shared helpers for the test suites of jabcode's
// internal packages. Its main job is locating the single repository-wide
// testdata/ directory so fixtures (golden vectors, reference images) live in one
// place instead of being copied into each package.
package testutil

import (
	"os"
	"path/filepath"
	"runtime"
)

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
