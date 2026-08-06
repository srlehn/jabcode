//go:build !js

package detect

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/wire"
)

// ldpcParityPlan builds the device plan for one code from the same cached
// parity-check matrix the host decoder uses, so a divergence is the corrector's
// and never the matrix's.
func ldpcParityPlan(t *testing.T, wc, wr, grossSub, blocks int) gpuLDPCPlan {
	t.Helper()
	layout, ok := ecc.ParityRows(wc, wr, grossSub, wire.ISO23634)
	if !ok {
		t.Fatalf("no parity rows for wc=%d wr=%d gross=%d", wc, wr, grossSub)
	}
	return gpuLDPCPlan{
		rows:      layout.Rows,
		rowDegree: layout.Degree,
		length:    grossSub,
		height:    layout.Height,
		rank:      layout.Rank,
		net:       grossSub * (wr - wc) / wr,
		blocks:    blocks,
	}
}

// TestGPULDPCHardParity pins the device corrector against the host decoder on
// codewords carrying a known number of injected errors. Hard LDPC has no
// payload integrity check underneath it, so a corrupted decode returns a
// plausible payload with no error: the message bits are compared in full rather
// than through a success count, and the syndrome verdict must agree too.
func TestGPULDPCHardParity(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU LDPC device: %v", err)
		}
	})

	codes := []struct {
		name   string
		wc, wr int
		net    int
	}{
		{name: "wc2wr4", wc: 2, wr: 4, net: 128},
		{name: "wc3wr6", wc: 3, wr: 6, net: 192},
		{name: "wc4wr8", wc: 4, wr: 8, net: 256},
	}
	for _, code := range codes {
		for _, errors := range []int{0, 1, 3} {
			t.Run(code.name, func(t *testing.T) {
				message := make([]byte, code.net)
				source := rand.New(rand.NewPCG(uint64(code.wc), uint64(errors)+1))
				for i := range message {
					message[i] = byte(source.UintN(2))
				}
				codeword := ecc.EncodeLDPC(message, code.wc, code.wr)
				if len(codeword) == 0 {
					t.Skipf("no codeword for wc=%d wr=%d", code.wc, code.wr)
				}
				for i := range errors {
					at := int(source.UintN(uint(len(codeword))))
					codeword[at] ^= 1
					_ = i
				}

				host := append([]byte(nil), codeword...)
				wantDec, wantOK := ecc.DecodeLDPCHard(host, code.wc, code.wr)

				plan := ldpcParityPlan(t, code.wc, code.wr, len(codeword), 1)
				if plan.length > gpuLDPCMaxSub {
					t.Skipf("codeword %d exceeds the device sub-block bound", plan.length)
				}
				gotDec, gotOK, err := resident.CorrectLDPCHard(plan, codeword)
				if err != nil {
					t.Fatalf("device LDPC correction: %v", err)
				}
				if gotOK != wantOK {
					t.Fatalf("%d errors: device syndrome ok=%t, host ok=%t", errors, gotOK, wantOK)
				}
				if len(gotDec) < len(wantDec) {
					t.Fatalf("device returned %d message bits, host %d", len(gotDec), len(wantDec))
				}
				if !bytes.Equal(gotDec[:len(wantDec)], wantDec) {
					differing := 0
					for i := range wantDec {
						if gotDec[i] != wantDec[i] {
							differing++
						}
					}
					t.Fatalf("%d errors: %d of %d message bits differ from the host decoder",
						errors, differing, len(wantDec))
				}
			})
		}
	}
}
