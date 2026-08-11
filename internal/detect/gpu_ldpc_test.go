//go:build !js

package detect

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
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

// TestGPULDPCSoftGraphFitsColumnSlots pins the format property the retry's
// resident reverse graph uses: payload rows are compact and every variable has
// exactly wc edges, within the ten-slot legal maximum.
func TestGPULDPCSoftGraphFitsColumnSlots(t *testing.T) {
	for _, variant := range []wire.Variant{wire.ISO23634, wire.CurrentC, wire.BSI, wire.PreV2C} {
		for _, code := range []struct {
			wc, wr int
		}{
			{3, 4},
			{3, 6},
			{7, 10},
			{10, 11},
		} {
			for _, rowsPerSet := range []int{1, 2, 6, 37, 2699 / code.wr} {
				length := rowsPerSet * code.wr
				rows, ok := ecc.ParityRows(code.wc, code.wr, length, variant)
				if !ok {
					t.Fatalf("variant=%d wc=%d wr=%d length=%d: no parity rows",
						variant, code.wc, code.wr, length)
				}
				edges, err := gpuLDPCSoftEdgesOf(rows.Rows, rows.Degree, rows.Height, length)
				if err != nil {
					t.Fatalf("variant=%d wc=%d wr=%d length=%d: %v",
						variant, code.wc, code.wr, length, err)
				}
				if edges != length*code.wc {
					t.Fatalf("variant=%d wc=%d wr=%d length=%d: %d edges, want %d",
						variant, code.wc, code.wr, length, edges, length*code.wc)
				}
				counts := make([]int, length)
				for _, column := range rows.Rows[:edges] {
					counts[column]++
				}
				for column, count := range counts {
					if count != code.wc {
						t.Fatalf("variant=%d wc=%d wr=%d length=%d: column %d has %d edges",
							variant, code.wc, code.wr, length, column, count)
					}
				}
			}
		}
	}
}

func TestGPULDPCSoftResidentBound(t *testing.T) {
	const wantBytes = 8_177_712
	if gpuLDPCSoftRetainedBytes != wantBytes {
		t.Fatalf("soft retry retains %d bytes, want %d", gpuLDPCSoftRetainedBytes, wantBytes)
	}
	if gpuLDPCSoftReliabilityIndirectOffset != 0 ||
		gpuLDPCSoftGraphIndirectOffset != 12 ||
		gpuLDPCSoftCorrectionIndirectOffset != 24 ||
		gpuPayloadClassificationIndirectOffset != 36 {
		t.Fatalf("indirect offsets are %d, %d, %d, %d",
			gpuLDPCSoftReliabilityIndirectOffset,
			gpuLDPCSoftGraphIndirectOffset,
			gpuLDPCSoftCorrectionIndirectOffset,
			gpuPayloadClassificationIndirectOffset,
		)
	}
}

// metadataParityPlan builds the device plan for one primary-metadata part. The
// metadata codes take HardBlockSplit's wr <= 3 branch, which **discards the
// caller's wc**: it sets WC to 2 unless the net length exceeds 36. The host
// decoder then builds its parity-check matrix from that rebound value, so a
// plan built from the wc the caller passed would be solving a different system.
func metadataParityPlan(t *testing.T, wc, length int) gpuLDPCPlan {
	t.Helper()
	split := ecc.HardBlockSplit(length, wc, 0)
	rows, ok := ecc.ParityRows(split.WC, 0, split.GrossSub, wire.ISO23634)
	if !ok {
		t.Fatalf("no parity rows for wc=%d wr=0 gross=%d", split.WC, split.GrossSub)
	}
	t.Logf(
		"metadata length %d: split wc=%d gross=%d net=%d blocks=%d, rows height=%d rank=%d degree=%d",
		length, split.WC, split.GrossSub, split.NetSub, split.Blocks,
		rows.Height, rows.Rank, rows.Degree,
	)
	return gpuLDPCPlan{
		rows:      rows.Rows,
		rowDegree: rows.Degree,
		length:    split.GrossSub,
		height:    rows.Height,
		rank:      rows.Rank,
		net:       split.NetSub,
		blocks:    split.Blocks,
	}
}

// TestGPULDPCHardMetadataParity pins the device corrector on the two
// primary-metadata instances, which is what the metadata walk has to run on the
// device. They are unlike every payload code the corrector was built for: Part I
// is six bits and falls under the kernel's single-flip branch, Part II is
// thirty-eight and does not, and both come out of HardBlockSplit's short-code
// path rather than its wr-driven one.
//
// The input is arbitrary bits rather than a valid codeword on purpose. Metadata
// is read off a capture, so the corrector's real input is frequently
// uncorrectable, and the two sides have to agree on those too - including on the
// syndrome verdict, since hard LDPC has nothing underneath it to catch a
// plausible wrong answer. Part I is exhaustive over all 64 inputs.
func TestGPULDPCHardMetadataParity(t *testing.T) {
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

	parts := []struct {
		name   string
		wc     int
		length int
	}{
		// The wc values are the ones DecodePrimaryMetadataPartI and PartII pass,
		// which is 3 below 36 bits and 4 above it.
		{name: "part I", wc: 3, length: spec.PrimaryMetadataPart1Length},
		{name: "part II", wc: 4, length: spec.PrimaryMetadataPart2Length},
	}
	for _, part := range parts {
		t.Run(part.name, func(t *testing.T) {
			plan := metadataParityPlan(t, part.wc, part.length)
			if !plan.valid() {
				t.Fatalf("metadata plan is out of the device corrector's range: %+v", plan)
			}
			vectors := metadataTestVectors(part.length)
			for index, bits := range vectors {
				host := append([]byte(nil), bits...)
				wantDec, wantOK := ecc.DecodeLDPCHardVariant(host, part.wc, 0, wire.ISO23634)
				gotDec, gotOK, err := resident.CorrectLDPCHard(plan, append([]byte(nil), bits...))
				if err != nil {
					t.Fatalf("vector %d: device LDPC correction: %v", index, err)
				}
				if gotOK != wantOK {
					t.Fatalf("vector %d (%v): device syndrome ok=%t, host ok=%t",
						index, bits, gotOK, wantOK)
				}
				if len(gotDec) < len(wantDec) {
					t.Fatalf("vector %d: device returned %d message bits, host %d",
						index, len(gotDec), len(wantDec))
				}
				if !bytes.Equal(gotDec[:len(wantDec)], wantDec) {
					t.Fatalf("vector %d (%v): device message %v, host %v",
						index, bits, gotDec[:len(wantDec)], wantDec)
				}
			}
			t.Logf("%d vectors agree bit for bit", len(vectors))
		})
	}
}

// metadataTestVectors enumerates every input for a part short enough to
// enumerate and samples one deterministically otherwise, always including the
// two constant vectors because they are the degenerate syndromes.
func metadataTestVectors(length int) [][]byte {
	if length <= 12 {
		vectors := make([][]byte, 0, 1<<length)
		for value := range 1 << length {
			bits := make([]byte, length)
			for i := range bits {
				bits[i] = byte((value >> (length - 1 - i)) & 1)
			}
			vectors = append(vectors, bits)
		}
		return vectors
	}
	vectors := [][]byte{make([]byte, length), make([]byte, length)}
	for i := range vectors[1] {
		vectors[1][i] = 1
	}
	source := rand.New(rand.NewPCG(uint64(length), 0x5eed))
	for range 256 {
		bits := make([]byte, length)
		for i := range bits {
			bits[i] = byte(source.UintN(2))
		}
		vectors = append(vectors, bits)
	}
	return vectors
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
