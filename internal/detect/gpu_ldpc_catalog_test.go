//go:build !js

package detect

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/ldpccatalog"
	"github.com/srlehn/jabcode/internal/wire"
)

// gpuLDPCCatalogKeys enumerates legal catalog keys, keeping every stride-th
// capacity of each row and column weight pair. A stride of one is the whole key
// space. The largest capacity of each row weight is always kept, because a
// directory read that runs off the end of a block lands there first.
func gpuLDPCCatalogKeys(stride int) [][3]int {
	if stride < 1 {
		stride = 1
	}
	var keys [][3]int
	for wr := ldpccatalog.MinRowWeight; wr <= ldpccatalog.MaxRowWeight; wr++ {
		last := ldpccatalog.MaxCapacity / wr
		for wc := ldpccatalog.MinColWeight; wc < wr; wc++ {
			for index := 1; index <= last; index++ {
				if index%stride != 0 && index != 1 && index != last {
					continue
				}
				keys = append(keys, [3]int{wc, wr, index * wr})
			}
		}
	}
	return keys
}

// checkGPULDPCCatalogKeys builds each key's parity matrix on the device and
// holds it to the host reduction.
func checkGPULDPCCatalogKeys(t *testing.T, keys [][3]int) {
	t.Helper()
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
			t.Errorf("close GPU catalog test device: %v", err)
		}
	})

	generators := 0
	for _, generator := range []ldpccatalog.Generator{
		ldpccatalog.GeneratorISO, ldpccatalog.GeneratorLCG,
	} {
		if !ldpccatalog.Wellformed(generator) {
			continue
		}
		generators++
		variant := generator.Variant()
		checked := 0
		for _, key := range keys {
			wc, wr, capacity := key[0], key[1], key[2]
			slot, ok := ldpccatalog.Slot(wc, wr, capacity)
			if !ok {
				t.Fatalf("wc=%d wr=%d capacity=%d is not a legal key", wc, wr, capacity)
			}
			name := fmt.Sprintf("generator=%d/wc=%d/wr=%d/capacity=%d", generator, wc, wr, capacity)
			if err := compareGPULDPCCatalogKey(resident, variant, wc, wr, capacity); err != nil {
				t.Fatalf("%s (slot %d): %v", name, slot, err)
			}
			checked++
		}
		t.Logf("generator %d: %d keys build on the device and match the host reduction",
			generator, checked)
	}
	if generators == 0 {
		t.Fatal("no pivot catalog is compiled in, so this build proves nothing")
	}
}

func compareGPULDPCCatalogKey(
	resident *gpuResidentBinarizer,
	variant wire.Variant,
	wc, wr, capacity int,
) error {
	layout := ecc.HardBlockLayout{
		WC:       wc,
		WR:       wr,
		Pg:       capacity,
		Pn:       capacity * (wr - wc) / wr,
		GrossSub: capacity,
		NetSub:   capacity * (wr - wc) / wr,
		Blocks:   1,
		Uniform:  true,
	}
	rows, control, _, err := resident.buildLDPCMatrix(layout, variant, false)
	if err != nil {
		return fmt.Errorf("build matrix: %w", err)
	}
	word := func(index int) int {
		return int(binary.LittleEndian.Uint32(control[index*4:]))
	}
	if admission := word(gpuLDPCParamAdmission); admission != 0 {
		return fmt.Errorf("device refused the key, admission %d", admission)
	}
	want, ok := ecc.ParityRows(wc, wr, capacity, variant)
	if !ok {
		return fmt.Errorf("host reduction produced no matrix")
	}
	if got := word(gpuLDPCParamHeight); got != want.Height {
		return fmt.Errorf("height %d, want %d", got, want.Height)
	}
	if got := word(gpuLDPCParamRowDegree); got != want.Degree {
		return fmt.Errorf("row degree %d, want %d", got, want.Degree)
	}
	if got := word(gpuLDPCParamRank); got != want.Rank {
		return fmt.Errorf("rank %d, want %d", got, want.Rank)
	}
	for at, column := range want.Rows {
		if got := binary.LittleEndian.Uint32(rows[at*4:]); got != column {
			return fmt.Errorf("row word %d is %d, want %d", at, got, column)
		}
	}
	return nil
}

// TestGPULDPCMatrixCatalogKeyWiring holds the device's catalog lookup to the
// host reduction across the key space rather than across a fixture list.
//
// Content parity on a few shapes is a different question from this one. The
// device derives its own slot from the payload shape, adds the generator's base
// offset and reads a packed record area; a drifted base or slot stride finds a
// well-formed record belonging to another key, and emits a parity matrix that
// is merely plausible. Hard LDPC has no payload integrity check underneath, so
// that decodes into silent corruption. Sweeping every row and column weight
// across the capacity range puts every directory block and both ends of each
// one under the comparison. The whole key space is the same test under
// jabcatalog_exhaustive.
func TestGPULDPCMatrixCatalogKeyWiring(t *testing.T) {
	checkGPULDPCCatalogKeys(t, gpuLDPCCatalogKeys(64))
}

// TestGPULDPCMatrixRefusesKeysPastTheCatalog pins the other half of the bounds
// test. The split that feeds this builder holds its uniform blocks under 2700
// bits but not its trailing one, which is the uniform length plus a remainder
// and so runs past MaxCapacity on large symbols. Both routes refuse those
// shapes before the lookup; if the device stopped doing so it would index a
// directory it does not have, and read whatever follows as a transcript.
func TestGPULDPCMatrixRefusesKeysPastTheCatalog(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
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
			t.Errorf("close GPU catalog test device: %v", err)
		}
	})
	for _, key := range [][3]int{
		{3, 4, (ldpccatalog.MaxCapacity/4 + 1) * 4},
		{10, 11, (ldpccatalog.MaxCapacity/11 + 1) * 11},
		{3, 4, 6},
		{3, 3, 12},
		{3, 12, 12},
	} {
		wc, wr, capacity := key[0], key[1], key[2]
		if _, ok := ldpccatalog.Slot(wc, wr, capacity); ok {
			t.Fatalf("wc=%d wr=%d capacity=%d is a legal key, so it proves nothing here",
				wc, wr, capacity)
		}
		err := compareGPULDPCCatalogKey(resident, wire.ISO23634, wc, wr, capacity)
		if err == nil {
			t.Fatalf("wc=%d wr=%d capacity=%d was built from the catalog", wc, wr, capacity)
		}
		if !strings.Contains(err.Error(), "device refused the key") {
			t.Fatalf("wc=%d wr=%d capacity=%d: %v, want a refusal", wc, wr, capacity, err)
		}
	}
}

// TestGPULDPCMatrixCacheAnswersASecondBuild pins the reuse the catalog design
// promises. Expanding a transcript is the only work the lookup still does, so a
// second request for the same key - the ordinary case across the levels of one
// image and across later images in a session - has to be answered from the
// resident key cache rather than replayed, and a different key has to replace
// it rather than be answered by it.
func TestGPULDPCMatrixCacheAnswersASecondBuild(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
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
			t.Errorf("close GPU catalog cache device: %v", err)
		}
	})
	layoutOf := func(capacity int) ecc.HardBlockLayout {
		const wc, wr = 3, 4
		return ecc.HardBlockLayout{
			WC: wc, WR: wr,
			Pg: capacity, Pn: capacity * (wr - wc) / wr,
			GrossSub: capacity, NetSub: capacity * (wr - wc) / wr,
			Blocks:  1,
			Uniform: true,
		}
	}
	first := layoutOf(512)
	second := layoutOf(1024)

	rows, control, cache, err := resident.buildLDPCMatrix(first, wire.ISO23634, false)
	if err != nil {
		t.Fatalf("first build: %v", err)
	}
	assertGPULDPCCacheKey(t, cache, first)

	// Identical output proves nothing on its own, because a rebuild produces it
	// too. The row area is overwritten between the two builds instead: a cache
	// hit republishes the control and leaves the rows alone, so rows that come
	// back rebuilt are a cache that did not answer.
	poisonGPULDPCRows(t, resident)
	replayRows, replayControl, replayCache, err := resident.buildLDPCMatrix(
		first, wire.ISO23634, true,
	)
	if err != nil {
		t.Fatalf("second build of the same key: %v", err)
	}
	assertGPULDPCCacheKey(t, replayCache, first)
	if !bytes.Equal(replayControl, control) {
		t.Fatal("the cached build published different control")
	}
	// Only the emitted region is meaningful: a build writes height by degree
	// words and leaves the rest of the area as it found it.
	emitted := int(binary.LittleEndian.Uint32(control[gpuLDPCParamHeight*4:])) *
		int(binary.LittleEndian.Uint32(control[gpuLDPCParamRowDegree*4:]))
	if emitted <= 0 {
		t.Fatalf("the first build emitted %d row words", emitted)
	}
	for at := range emitted {
		if got := binary.LittleEndian.Uint32(replayRows[at*4:]); got != 0xdeadbeef {
			t.Fatalf("row word %d was rebuilt as %d, so the resident key cache did not answer",
				at, got)
		}
	}
	_ = rows

	// A different key has to rebuild rather than be answered by the cache.
	otherRows, _, otherCache, err := resident.buildLDPCMatrix(second, wire.ISO23634, true)
	if err != nil {
		t.Fatalf("build of another key: %v", err)
	}
	assertGPULDPCCacheKey(t, otherCache, second)
	want, ok := ecc.ParityRows(second.WC, second.WR, second.GrossSub, wire.ISO23634)
	if !ok {
		t.Fatal("host reduction produced no matrix for the second key")
	}
	for at, column := range want.Rows {
		if got := binary.LittleEndian.Uint32(otherRows[at*4:]); got != column {
			t.Fatalf("second key row word %d = %d, want %d", at, got, column)
		}
	}
}

// poisonGPULDPCRows overwrites the resident row area so a later build that does
// not re-emit it is visible.
func poisonGPULDPCRows(t *testing.T, resident *gpuResidentBinarizer) {
	t.Helper()
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		t.Fatalf("create row poison recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Fill(resident.ldpcRows, 0, 2*gpuLDPCRowWords*4, 0xdeadbeef); err != nil {
		t.Fatalf("poison the resident rows: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run the row poison: %v", err)
	}
}

// assertGPULDPCCacheKey checks the resident cache describes the key just built.
func assertGPULDPCCacheKey(t *testing.T, cache []byte, layout ecc.HardBlockLayout) {
	t.Helper()
	word := func(index int) int {
		return int(binary.LittleEndian.Uint32(cache[index*4:]))
	}
	const (
		cacheValid = iota
		cacheLength
		cacheWC
		cacheWR
	)
	if word(cacheValid) == 0 {
		t.Fatal("the resident matrix cache is empty after a build")
	}
	if got := word(cacheLength); got != layout.GrossSub {
		t.Fatalf("cached length = %d, want %d", got, layout.GrossSub)
	}
	if got := word(cacheWC); got != layout.WC {
		t.Fatalf("cached column weight = %d, want %d", got, layout.WC)
	}
	if got := word(cacheWR); got != layout.WR {
		t.Fatalf("cached row weight = %d, want %d", got, layout.WR)
	}
}
