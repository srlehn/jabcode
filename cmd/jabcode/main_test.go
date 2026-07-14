package main

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

type closeErrorWriter struct {
	bytes.Buffer
	err error
}

func (w *closeErrorWriter) Close() error { return w.err }

func TestEncodeUsageDescribesLiteralInput(t *testing.T) {
	var out bytes.Buffer
	encodeUsage(&out)
	want := "-i, --input string        literal input text; omit it to read stdin"
	if !strings.Contains(out.String(), want) {
		t.Fatalf("encode usage missing %q:\n%s", want, out.String())
	}
}

func TestUsageMarksISOProfileExperimental(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage func(io.Writer)
	}{
		{"encode", encodeUsage},
		{"decode", decodeUsage},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			tc.usage(&out)
			usage := strings.ToLower(out.String())
			if !strings.Contains(usage, "iso") || !strings.Contains(usage, "experimental") {
				t.Fatalf("usage does not mark the ISO profile experimental:\n%s", out.String())
			}
		})
	}
}

func TestParseDecodeOnly(t *testing.T) {
	for _, tc := range []struct {
		value   string
		variant wire.Variant
	}{
		{"ISO-23634", wire.ISO23634},
		{"hc", wire.ISOHighColor},
		{"high-color", wire.ISOHighColor},
		{"current-c", wire.CurrentC},
		{"bsi", wire.BSI},
		{"pre-v2-c", wire.PreV2C},
	} {
		variant, err := parseDecodeOnly(tc.value)
		if err != nil {
			t.Errorf("parseDecodeOnly(%q): %v", tc.value, err)
			continue
		}
		if variant != tc.variant {
			t.Errorf("parseDecodeOnly(%q) = %d, want %d", tc.value, variant, tc.variant)
		}
	}
	for _, obsolete := range []string{"legacy", "c", "compat", "c-reference"} {
		if _, err := parseDecodeOnly(obsolete); err == nil {
			t.Errorf("parseDecodeOnly accepted obsolete alias %q", obsolete)
		}
	}
	if _, err := parseDecodeOnly("future"); err == nil {
		t.Error("parseDecodeOnly accepted an unknown format")
	}
}

func TestEncodeInputExplicitEmptyLiteral(t *testing.T) {
	data, err := encodeInput("", true)
	if err != nil {
		t.Fatalf("encodeInput: %v", err)
	}
	if len(data) != 0 {
		t.Fatalf("encodeInput returned %q, want empty literal", data)
	}
}

func TestEncodeInputLiteral(t *testing.T) {
	data, err := encodeInput("hello", true)
	if err != nil {
		t.Fatalf("encodeInput: %v", err)
	}
	if string(data) != "hello" {
		t.Fatalf("encodeInput returned %q, want hello", data)
	}
}

func TestEncodeInputDashLiteral(t *testing.T) {
	data, err := encodeInput("-", true)
	if err != nil {
		t.Fatalf("encodeInput: %v", err)
	}
	if string(data) != "-" {
		t.Fatalf("encodeInput returned %q, want dash literal", data)
	}
}

func TestEncodeInputDefaultReadsStdin(t *testing.T) {
	oldStdin := os.Stdin
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	t.Cleanup(func() {
		os.Stdin = oldStdin
		r.Close()
	})
	os.Stdin = r
	if _, err := w.WriteString("from stdin"); err != nil {
		t.Fatalf("write pipe: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close pipe writer: %v", err)
	}

	data, err := encodeInput("", false)
	if err != nil {
		t.Fatalf("encodeInput: %v", err)
	}
	if string(data) != "from stdin" {
		t.Fatalf("encodeInput returned %q, want stdin payload", data)
	}
}

func TestWritePayloadDashWritesStdout(t *testing.T) {
	out := captureStdout(t, func() error {
		return writePayload("-", []byte("payload"))
	})
	if string(out) != "payload" {
		t.Fatalf("writePayload wrote %q, want payload", out)
	}
}

func TestWritePNGDashWritesStdout(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 12, G: 34, B: 56, A: 255})

	out := captureStdout(t, func() error {
		return writePNG("-", img)
	})
	decoded, err := png.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode stdout png: %v", err)
	}
	if got := decoded.Bounds().Size(); got != image.Pt(1, 1) {
		t.Fatalf("decoded PNG size = %v, want 1x1", got)
	}
}

func TestWritePNGFileReturnsCloseError(t *testing.T) {
	want := errors.New("close failed")
	w := &closeErrorWriter{err: want}
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	if err := writePNGFile(w, img, "out.png"); !errors.Is(err, want) {
		t.Fatalf("writePNGFile error = %v, want close error", err)
	}
}

func captureStdout(t *testing.T, fn func() error) []byte {
	t.Helper()
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	callErr := fn()
	closeErr := w.Close()
	os.Stdout = oldStdout
	defer r.Close()
	if callErr != nil {
		t.Fatalf("write stdout: %v", callErr)
	}
	if closeErr != nil {
		t.Fatalf("close stdout writer: %v", closeErr)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	return out
}
