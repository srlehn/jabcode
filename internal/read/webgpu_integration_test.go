//go:build js

package read

import (
	"bytes"
	"syscall/js"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

func TestAutomaticWebGPUPublicDecode(t *testing.T) {
	navigator := js.Global().Get("navigator")
	if !navigator.Truthy() || !navigator.Get("gpu").Truthy() {
		t.Skip("no navigator.gpu in this runtime")
	}

	payload := []byte("automatic WebGPU public decode")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 64, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if b := img.Bounds(); b.Dx()*b.Dy() < 1024*1024 {
		t.Fatalf("test image %v is below the automatic GPU threshold", b)
	}
	gpuSession, err := detect.NewAutomaticGPUDecodeSession(core.BitmapFromImage(img), len(pyramidLevels(img)))
	if err != nil || gpuSession == nil {
		t.Fatalf("automatic WebGPU session unavailable: %v", err)
	}
	defer gpuSession.Close()

	cpu, _, _, ok := decodePyramidCapabilitiesWithGPU(
		newPyramid(img), nil, compiledCapabilities(), nil,
	)
	if !ok || cpu == nil {
		t.Fatal("forced CPU decode failed")
	}
	want := messageTransmission(cpu)

	for i := 0; i < 2; i++ {
		got, err := Decode(img)
		if err != nil {
			t.Fatalf("automatic Decode run %d: %v", i+1, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("automatic Decode run %d = %q, want %q", i+1, got, want)
		}
	}
}
