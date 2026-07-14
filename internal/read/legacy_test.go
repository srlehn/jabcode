package read

import (
	"image"
	"image/png"
	"os"
	"testing"

	"github.com/srlehn/jabcode/internal/testutil"
)

func loadLegacyCReferenceFixture(t *testing.T, name string) image.Image {
	t.Helper()
	f, err := os.Open(testutil.TestdataPath(name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}
