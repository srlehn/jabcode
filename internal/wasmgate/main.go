// Command wasmgate executes the public fixed-plan and stream path. It exists
// only as the regular-Go WebAssembly execution target for the repository's
// host-side gate.
package main

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"time"

	"github.com/srlehn/jabcode"
)

func main() {
	fmt.Println("WASM_READY")
	payload := opaqueByteCorpus()
	plan, err := jabcode.NewOpaquePlan(image.Pt(8, 8), jabcode.WithModuleSize(6))
	if err != nil {
		fail("NewOpaquePlan: %v", err)
	}
	img, err := plan.Encode(payload)
	if err != nil {
		fail("Encode: %v", err)
	}

	stream := jabcode.NewStream()
	firstStart := time.Now()
	first, err := stream.DecodeMessage(img)
	firstElapsed := time.Since(firstStart)
	if err != nil || !bytes.Equal(first.Data, payload) {
		fail("first DecodeMessage = %d bytes, %v", len(first.Data), err)
	}
	lockedStart := time.Now()
	locked, err := stream.DecodeMessage(img)
	lockedElapsed := time.Since(lockedStart)
	if err != nil || !bytes.Equal(locked.Data, payload) {
		fail("locked DecodeMessage = %d bytes, %v", len(locked.Data), err)
	}
	blank := image.NewNRGBA(image.Rectangle{Max: plan.ImageDimensions()})
	noSymbolStart := time.Now()
	_, err = stream.DecodeMessage(blank)
	noSymbolElapsed := time.Since(noSymbolStart)
	if err == nil {
		fail("blank frame decoded as a symbol")
	}
	fmt.Printf(
		"WASM_METRIC first_decode_ns=%d locked_replay_ns=%d no_symbol_ns=%d capacity=%d image_width=%d image_height=%d\n",
		firstElapsed.Nanoseconds(), lockedElapsed.Nanoseconds(), noSymbolElapsed.Nanoseconds(),
		plan.Capacity(), plan.ImageDimensions().X, plan.ImageDimensions().Y,
	)
}

func opaqueByteCorpus() []byte {
	payload := make([]byte, 0, 320)
	for value := range 256 {
		payload = append(payload, byte(value))
	}
	return append(payload,
		0, '\\', '\\', 0,
		']', 'j', '1', ']', 'j', '4', ']', 'j', '5',
		'\\', '0', '0', '0', '0', '2', '6',
		29, 30, 4,
	)
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "wasmgate: "+format+"\n", args...)
	os.Exit(1)
}
