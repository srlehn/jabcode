package main

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"testing"
)

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
