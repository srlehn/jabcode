package detect

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// prunePatterns builds one candidate per type with the given FoundCounts, at
// positions far enough apart that grouping never merges them. A zero count
// means the type produced nothing at all.
func prunePatterns(counts [4]int) ([]FinderPattern, int) {
	fps := make([]FinderPattern, maxFinderPatterns)
	total := 0
	for typ, n := range counts {
		if n == 0 {
			continue
		}
		fps[total] = FinderPattern{
			Typ:        typ,
			Center:     core.Pt(float64(100+200*typ), float64(100+200*typ)),
			ModuleSize: 8,
			FoundCount: n,
		}
		total++
	}
	return fps, total
}

// TestOutvotedPruneKeepsTheSelectionRecoverable is written from the measured
// case: on an oblique capture the one scan direction holding all four finder
// types had counts 5/4/8/14, and pruning everything under half of 14 removed
// two of them, including a corner sitting exactly on the symbol. One absent
// type is interpolated from the other three; two is fatal, so that prune threw
// the direction away rather than cleaning it up.
func TestOutvotedPruneKeepsTheSelectionRecoverable(t *testing.T) {
	for _, tc := range []struct {
		name        string
		counts      [4]int
		wantMissing int
		wantKept    [4]bool
	}{{
		// The measured direction-45 selection. Only the weaker of the two
		// outvoted types may go: dropping both leaves nothing to interpolate
		// from.
		name:        "two outvoted, only the weaker goes",
		counts:      [4]int{5, 4, 8, 14},
		wantMissing: 1,
		wantKept:    [4]bool{true, false, true, true},
	}, {
		// The rule changes nothing where the prune was already recoverable.
		name:        "one outvoted",
		counts:      [4]int{2, 20, 18, 16},
		wantMissing: 1,
		wantKept:    [4]bool{false, true, true, true},
	}, {
		// A type that was never found has already spent the one absence, so no
		// prune can follow it.
		name:        "already missing one",
		counts:      [4]int{0, 3, 12, 14},
		wantMissing: 1,
		wantKept:    [4]bool{false, true, true, true},
	}, {
		name:        "nothing outvoted",
		counts:      [4]int{9, 10, 11, 12},
		wantMissing: 0,
		wantKept:    [4]bool{true, true, true, true},
	}, {
		// Equal counts are ordered by type, so which one goes is fixed rather
		// than left to map or sort order.
		name:        "equal outvoted counts",
		counts:      [4]int{3, 3, 14, 14},
		wantMissing: 1,
		wantKept:    [4]bool{false, true, true, true},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			fps, total := prunePatterns(tc.counts)
			var st FinderFamilyScanStats
			var d PrimaryDetector
			missing := d.selectBestPatternsFor(fps, total, make([]int, 4), &st, nil)

			if missing != tc.wantMissing {
				t.Errorf("missing=%d, want %d (selected %v)", missing, tc.wantMissing, st.Selected)
			}
			for typ, want := range tc.wantKept {
				if got := fps[typ].FoundCount > 0; got != want {
					t.Errorf("type %d kept=%v, want %v (counts %v, selected %v)",
						typ, got, want, tc.counts, st.Selected)
				}
			}
			// A prune that leaves two types absent has discarded the direction:
			// nothing downstream can rebuild a quad from two corners.
			if missing >= 2 {
				t.Errorf("the prune left %d types absent, which no interpolation recovers", missing)
			}
		})
	}
}

// TestOutvotedPruneStillRunsOnNoise keeps the rule from becoming "never prune".
// A type confirmed three times against a maximum of forty is what the prune
// exists for, and it must still go when the selection can spare it.
//
// Three is the floor: a candidate under it never enters a group at all, so a
// smaller count would be removed before the prune and would test nothing.
func TestOutvotedPruneStillRunsOnNoise(t *testing.T) {
	fps, total := prunePatterns([4]int{3, 40, 38, 36})
	var st FinderFamilyScanStats
	var d PrimaryDetector
	if missing := d.selectBestPatternsFor(fps, total, make([]int, 4), &st, nil); missing != 1 {
		t.Fatalf("missing=%d, want the noise type pruned", missing)
	}
	if fps[0].FoundCount != 0 {
		t.Errorf("the outvoted type survived: %v", st.Selected)
	}
}

// TestPrintPassSkipsTheOutvotedPrune pins the existing exemption, which the
// reordering must not quietly drop: colorant-plane misregistration degrades the
// corners asymmetrically, and those candidates already passed widened checks.
func TestPrintPassSkipsTheOutvotedPrune(t *testing.T) {
	fps, total := prunePatterns([4]int{3, 40, 38, 36})
	var st FinderFamilyScanStats
	d := PrimaryDetector{printPass: true}
	if missing := d.selectBestPatternsFor(fps, total, make([]int, 4), &st, nil); missing != 0 {
		t.Fatalf("missing=%d, want the print pass to keep every type", missing)
	}
}
