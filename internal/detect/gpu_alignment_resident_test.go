//go:build !js

package detect

import (
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/tables"
)

func TestResidentAlignmentShaderTablesMatchWireTables(t *testing.T) {
	gotNum := wgslUintArray(t, alignmentPrepareWGSL, "AP_NUM")
	if len(gotNum) != len(tables.APNum) {
		t.Fatalf("AP_NUM has %d entries, want %d", len(gotNum), len(tables.APNum))
	}
	for at, want := range tables.APNum {
		if gotNum[at] != want {
			t.Fatalf("AP_NUM[%d] = %d, want %d", at, gotNum[at], want)
		}
	}

	gotPos := wgslUintArray(t, alignmentPrepareWGSL, "AP_POS")
	if len(gotPos) != len(tables.APPos)*len(tables.APPos[0]) {
		t.Fatalf("AP_POS has %d entries, want %d", len(gotPos), len(tables.APPos)*len(tables.APPos[0]))
	}
	for version, row := range tables.APPos {
		for position, want := range row {
			at := version*len(row) + position
			if gotPos[at] != want {
				t.Fatalf("AP_POS[%d][%d] = %d, want %d", version, position, gotPos[at], want)
			}
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
