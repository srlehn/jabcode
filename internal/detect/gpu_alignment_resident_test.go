//go:build !js

package detect

import (
	"encoding/binary"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/tables"
)

// TestResidentAlignmentTableBufferMatchesWireTables holds the device's copy of
// the alignment tables to the host's. The shaders used to carry their own const
// copies, which both duplicated these numbers and read as zero under a runtime
// version index; the buffer is now the only device copy, so this is where a
// drift would show.
func TestResidentAlignmentTableBufferMatchesWireTables(t *testing.T) {
	raw := gpuAlignmentTableBytes()
	if len(raw) != gpuAlignTableWords*4 {
		t.Fatalf("alignment table is %d bytes, want %d", len(raw), gpuAlignTableWords*4)
	}
	word := func(index int) int {
		return int(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	for version, want := range tables.APNum {
		if got := word(version); got != want {
			t.Fatalf("pattern count for version %d = %d, want %d", version+1, got, want)
		}
	}
	for version, row := range tables.APPos {
		for position, want := range row {
			at := gpuAlignTablePos + version*gpuAlignTableStride + position
			if got := word(at); got != want {
				t.Fatalf("position %d of version %d = %d, want %d",
					position, version+1, got, want)
			}
		}
	}
	// The confirmation kernel reads the second position of a version through
	// the same buffer, at the offset its own helper computes.
	for version, row := range tables.APPos {
		at := gpuAlignTablePos + version*gpuAlignTableStride + 1
		if got := word(at); got != row[1] {
			t.Fatalf("second position of version %d = %d, want %d", version+1, got, row[1])
		}
	}
}

func TestResidentAlignmentShaderShapesMatchBuffers(t *testing.T) {
	wants := map[string]uint32{
		"RESULT_WORDS": gpuPrimaryResultWords,
		"RESULT_SLOTS": gpuPrimaryResultBatchSlots,
	}
	for name, want := range wants {
		if got := wgslUintConstant(t, alignmentPrepareWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
	if got := wgslUintConstant(t, alignmentRectsWGSL, "RECT_WORDS"); got != uint32(gpuAlignRectWords) {
		t.Errorf("RECT_WORDS = %d, want %d", got, gpuAlignRectWords)
	}
	if gpuAlignIndirectAttempt != gpuFinderGeometryAttemptIndirectOffset {
		t.Fatalf("alignment attempt offset = %d, want %d", gpuAlignIndirectAttempt, gpuFinderGeometryAttemptIndirectOffset)
	}
	if got := wgslUintConstant(t, alignmentPrepareWGSL, "INDIRECT_STATE"); got != uint32(gpuAlignIndirectState) {
		t.Fatalf("alignment state offset = %d, want %d", got, gpuAlignIndirectState)
	}
	if got := wgslUintConstant(t, alignmentConfirmWGSL, "INDIRECT_STATE"); got != uint32(gpuAlignIndirectState) {
		t.Fatalf("alignment confirmation state offset = %d, want %d", got, gpuAlignIndirectState)
	}
	if gpuAlignIndirectWords != gpuAlignIndirectConfirmedY+1 {
		t.Fatalf("alignment control has %d words, want %d", gpuAlignIndirectWords, gpuAlignIndirectConfirmedY+1)
	}
	if got := wgslUintConstant(t, alignmentConfirmWGSL, "CONFIRM_CANDIDATES"); got > gpuAlignMaxPositions {
		t.Fatalf("alignment confirmation needs %d candidates, capacity is %d", got, gpuAlignMaxPositions)
	}
}

func wgslUintArray(t *testing.T, source, name string) []int {
	t.Helper()
	start := strings.Index(source, "const "+name+":")
	if start < 0 {
		t.Fatalf("WGSL array %s is missing", name)
	}
	constructor := strings.Index(source[start:], "(")
	if constructor < 0 {
		t.Fatalf("WGSL array %s has no constructor", name)
	}
	constructor += start + 1
	end := strings.Index(source[constructor:], ");")
	if end < 0 {
		t.Fatalf("WGSL array %s is unterminated", name)
	}
	body := source[constructor : constructor+end]
	matches := regexp.MustCompile(`\b([0-9]+)u\b`).FindAllStringSubmatch(body, -1)
	out := make([]int, len(matches))
	for at, match := range matches {
		value, err := strconv.Atoi(match[1])
		if err != nil {
			t.Fatalf("parse WGSL array %s entry %q: %v", name, match[1], err)
		}
		out[at] = value
	}
	return out
}
