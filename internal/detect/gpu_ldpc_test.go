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

// ldpcParityPlan builds the device plan for one codeword through the same
// sub-block split and the same cached parity-check matrices the host decoder
// uses, so a divergence is the corrector's and never the code's.
func ldpcParityPlan(t *testing.T, wc, wr, length int) (gpuLDPCPlan, ecc.HardBlockLayout) {
	t.Helper()
	split := ecc.HardBlockSplit(length, wc, wr)
	rows, ok := ecc.ParityRows(wc, wr, split.GrossSub, wire.ISO23634)
	if !ok {
		t.Fatalf("no parity rows for wc=%d wr=%d gross=%d", wc, wr, split.GrossSub)
	}
	plan := gpuLDPCPlan{
		rows:      rows.Rows,
		rowDegree: rows.Degree,
		length:    split.GrossSub,
		height:    rows.Height,
		rank:      rows.Rank,
		net:       split.NetSub,
		blocks:    split.Blocks,
	}
	if split.Uniform {
		return plan, split
	}
	tailGross := split.TrailingGrossSub()
	tailRows, ok := ecc.ParityRows(wc, wr, tailGross, wire.ISO23634)
	if !ok {
		t.Fatalf("no trailing parity rows for wc=%d wr=%d gross=%d", wc, wr, tailGross)
	}
	plan.tailRows = tailRows.Rows
	plan.tailRowDegree = tailRows.Degree
	plan.tailLength = tailGross
	plan.tailHeight = tailRows.Height
	plan.tailRank = tailRows.Rank
	plan.tailNet = tailGross * (wr - wc) / wr
	return plan, split
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
		// split says whether the codeword divides into equal sub-blocks. The
		// long cases do not, which is the ordinary shape of a real symbol: the
		// trailing block absorbs the remainder and corrects under a matrix of
		// its own, and only these cases reach that path.
		uniform bool
	}{
		{name: "wc2wr4", wc: 2, wr: 4, net: 128, uniform: true},
		{name: "wc3wr6", wc: 3, wr: 6, net: 192, uniform: true},
		{name: "wc4wr8", wc: 4, wr: 8, net: 256, uniform: true},
		{name: "wc5wr11 split", wc: 5, wr: 11, net: 5298},
		{name: "wc4wr7 split", wc: 4, wr: 7, net: 3000},
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

				plan, split := ldpcParityPlan(t, code.wc, code.wr, len(codeword))
				if split.Uniform != code.uniform {
					t.Fatalf("codeword of %d bits split uniform=%t, want %t",
						len(codeword), split.Uniform, code.uniform)
				}
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
