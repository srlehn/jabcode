package jabcode

import (
	"bytes"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultEncodingIsExperimentalISO23634(t *testing.T) {
	payload := []byte("default ISO encoding")
	img, err := NewEncoder().Encode(payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := Decode(img)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if want := isoReaderTransmission(payload); !bytes.Equal(got, want) {
		t.Fatalf("Decode = %q, want %q", got, want)
	}
}

func TestISOEncodingRejectsReservedColorModes(t *testing.T) {
	for _, colors := range []int{16, 32, 64, 128, 256} {
		if _, err := NewEncoder(WithColors(colors)).Encode([]byte("reserved")); err == nil {
			t.Errorf("colors %d: expected ISO reserved-mode error", colors)
		}
	}
}

func declaredPackageNames(t *testing.T, files []string) map[string]bool {
	t.Helper()
	set := token.NewFileSet()
	names := make(map[string]bool)
	for _, name := range files {
		file, err := parser.ParseFile(set, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, decl := range file.Decls {
			switch decl := decl.(type) {
			case *ast.FuncDecl:
				if decl.Recv == nil {
					names[decl.Name.Name] = true
				}
			case *ast.GenDecl:
				for _, spec := range decl.Specs {
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						names[spec.Name.Name] = true
					case *ast.ValueSpec:
						for _, ident := range spec.Names {
							names[ident.Name] = true
						}
					}
				}
			}
		}
	}
	return names
}

func packageNamesForTags(t *testing.T, tags ...string) map[string]bool {
	t.Helper()
	ctx := build.Default
	ctx.BuildTags = tags
	pkg, err := ctx.ImportDir(".", build.IgnoreVendor)
	if err != nil {
		t.Fatal(err)
	}
	return declaredPackageNames(t, pkg.GoFiles)
}

func TestPublicSurfaceByBuildTags(t *testing.T) {
	testCases := []struct {
		name        string
		tags        []string
		wantProfile bool
	}{
		{name: "untagged"},
		{name: "high-color decoder", tags: []string{"jabcode_high_color"}},
		{name: "BSI decoder", tags: []string{"jabcode_bsi"}},
		{name: "historical-C decoder", tags: []string{"jabcode_legacy"}},
		{name: "all decoders", tags: []string{"jabcode_high_color", "jabcode_bsi", "jabcode_legacy"}},
		{name: "non-ISO encoder", tags: []string{"jabcode_non_iso_encode"}, wantProfile: true},
		{name: "encoder and all decoders", tags: []string{"jabcode_non_iso_encode", "jabcode_high_color", "jabcode_bsi", "jabcode_legacy"}, wantProfile: true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			names := packageNamesForTags(t, tc.tags...)
			for _, name := range []string{
				"Profile", "ProfileISO23634", "ProfileHighColor", "ProfileBSI", "WithProfile",
			} {
				if names[name] != tc.wantProfile {
					t.Errorf("%s presence = %v, want %v", name, names[name], tc.wantProfile)
				}
			}
			for _, name := range []string{
				"ProfileLegacy", "DecodeWithProfile",
				"NewStreamWithProfile", "ErrProfileUnavailable", "ErrProfileReadOnly",
			} {
				if names[name] {
					t.Errorf("package unexpectedly exports %s", name)
				}
			}
		})
	}
}

func TestNoBuildExportsLegacySelectors(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") ||
			strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		files = append(files, filepath.Clean(entry.Name()))
	}
	names := declaredPackageNames(t, files)
	for _, name := range []string{"ProfileLegacy"} {
		if names[name] {
			t.Errorf("production source exports gated selector %s", name)
		}
	}
}
