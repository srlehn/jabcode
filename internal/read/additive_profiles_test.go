//go:build jabcode_high_color && jabcode_legacy

package read

import (
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/testutil"
)

func TestAdditiveProfilesDecodeExistingHighColorSources(t *testing.T) {
	for _, tc := range []struct {
		file   string
		header string
	}{
		{"16c_ecc10_v15_lorem_ms4.png", "JAB high-colour capture test | colors=16"},
		{"256c_ecc10_v9_lorem_ms6.png", "JAB high-colour capture test | colors=256"},
	} {
		t.Run(tc.file, func(t *testing.T) {
			path := filepath.Join(testutil.TestdataPath("highcolor_capture"), "source", tc.file)
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			img, err := png.Decode(f)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Decode(img)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(string(got), tc.header) {
				t.Fatalf("Decode prefix = %q, want %q", got[:min(len(got), len(tc.header))], tc.header)
			}
		})
	}
}
