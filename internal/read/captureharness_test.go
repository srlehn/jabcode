//go:build jabharness

package read

import (
	"bytes"
	"fmt"
	"image"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/testutil"
)

// Capture-harness knobs.
const (
	// captureDirEnv points the fixture walk at a private or future capture set
	// (e.g. frames extracted from the live videos) without any code reference
	// to its location. Unset, the walk covers the repository's
	// testdata/highcolor_capture tree.
	captureDirEnv = "JABCAPTURE_DIR"
	// captureUpdateEnv, when set to a non-empty value, rewrites the committed
	// baseline from the current run instead of comparing against it. Advancing
	// the baseline is always a deliberate act reviewed like any other diff.
	captureUpdateEnv = "JABCAPTURE_UPDATE"
	// capturePerImageTimeout bounds one fixture's plain decode. The slowest
	// measured capture failure was 124 s under 6-way parallel load; the budget
	// is generous so only a hang, never a slow retry ladder, trips it.
	capturePerImageTimeout = 300 * time.Second
	// captureParallel bounds how many fixtures decode concurrently. Decode
	// fans out internally, so more workers only oversubscribe the box; the
	// per-fixture wall times are informational and never compared.
	captureParallel = 6
)

// captureBaseline is the committed row-by-row baseline the harness diffs
// against. It lives in the tracked central testdata tree NEXT TO, not inside,
// the fixture directory: the fixtures stay uncommitted until the user commits
// them personally, while the baseline is ordinary reviewed test data.
const captureBaseline = "highcolor_capture_baseline.tsv"

// captureClass is the payload-level outcome of one capture fixture.
type captureClass string

const (
	// captureOK: the payload is byte-correct for the fixture's colour count.
	captureOK captureClass = "ok"
	// captureOtherCode: a valid payload of a DIFFERENT colour count - the
	// decoder locked onto a neighbouring code (the A4 print sheet is a 2x3
	// grid, so oblique captures contain up to five neighbour codes). Neither
	// pass nor plain fail: orientation fixes are expected to surface these.
	captureOtherCode captureClass = "other-code"
	// captureCorrupt: err=nil with a payload matching no known capture
	// payload - the residual hard-LDPC hazard class (a miscorrection landing
	// in a different valid codeword), the worst outcome.
	captureCorrupt captureClass = "corrupt"
	// captureFail: no decode.
	captureFail captureClass = "fail"
)

// captureClassRank orders classes from worst to best so a baseline diff can
// tell a regression from an improvement: corrupt (silent wrong bytes) below
// fail, fail below other-code (detection worked, wrong neighbour), ok on top.
func captureClassRank(c captureClass) int {
	switch c {
	case captureCorrupt:
		return 0
	case captureFail:
		return 1
	case captureOtherCode:
		return 2
	case captureOK:
		return 3
	}
	return -1
}

// captureStageRank orders stage buckets by pipeline progress for the baseline
// diff. The buckets are the degradation harness's pipelineStage strings plus
// "timeout" (the per-image budget tripped; ranked below every real stage
// because nothing was attributed). Unknown strings rank lowest.
func captureStageRank(s string) int {
	switch s {
	case stageNoFinders.String():
		return 2
	case stageNoSideSize.String():
		return 3
	case stageNoSample.String():
		return 4
	case stageSampled.String():
		return 5
	case stageDecoded.String():
		return 6
	case "timeout":
		return 1
	}
	return 0
}

// captureRow is one fixture's measured outcome. Path, class and stage are the
// row's compared identity; the wall time and note are informational only (the
// time varies with machine and parallel load).
type captureRow struct {
	path  string // fixture path relative to the capture dir, slash-separated
	class captureClass
	stage string
	wall  time.Duration
	note  string
}

// TestCaptureHarness runs the public Decode over every real capture fixture
// and reports one row per fixture: payload class (ok / other-code / corrupt /
// fail), pipeline stage bucket, and wall time. The table is compared ROW BY
// ROW against the committed baseline: any row whose class or stage moves down
// fails the test; rows that move up are logged so the baseline can be advanced
// deliberately (JABCAPTURE_UPDATE=1 rewrites it). Ground-truth payloads come
// from the pixel-exact source fixtures, whose self-documenting headers pin the
// payload per colour count.
//
// The fixture tree is testdata/highcolor_capture (skipped cleanly when absent,
// because clones do not have it until the user commits it) or $JABCAPTURE_DIR;
// an overridden tree is measured and reported without a baseline comparison.
//
//	go test -tags jabharness -run TestCaptureHarness -timeout 40m -v ./internal/read
//
// 40m covers real runs comfortably (measured ~12 min); it cannot cover the
// pathological case of MANY images hitting the per-image budget - the floor
// there is ceil(79/6) batches x 300 s = 70 min before overhead, more with
// fewer cores or leaked timed-out decodes still running - so if a run ever
// nears the package timeout, rerun with -timeout 90m instead of trusting a
// truncated table.
func TestCaptureHarness(t *testing.T) {
	dir := os.Getenv(captureDirEnv)
	defaultDir := dir == ""
	if defaultDir {
		dir = testutil.TestdataPath("highcolor_capture")
	}
	fixtures := listCaptureFixtures(t, dir)
	known := captureGroundTruth(t, dir)
	for _, rel := range fixtures {
		colors, err := captureColorCount(rel)
		if err != nil {
			t.Fatalf("fixture %s: %v", rel, err)
		}
		if _, ok := known[colors]; !ok {
			t.Fatalf("fixture %s: no ground-truth payload for %d colours", rel, colors)
		}
	}

	rows := make([]captureRow, len(fixtures))
	idx := make(chan int)
	var wg sync.WaitGroup
	for range min(captureParallel, runtime.GOMAXPROCS(0)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range idx {
				rows[i] = measureCapture(t, dir, fixtures[i], known)
			}
		}()
	}
	for i := range fixtures {
		idx <- i
	}
	close(idx)
	wg.Wait()

	t.Logf("capture harness results:\n%s", formatCaptureReport(rows))

	baseline := testutil.TestdataPath(captureBaseline)
	switch {
	case os.Getenv(captureUpdateEnv) != "":
		if !defaultDir {
			t.Fatalf("%s only applies to the default fixture tree, not $%s", captureUpdateEnv, captureDirEnv)
		}
		if err := writeCaptureBaseline(baseline, rows); err != nil {
			t.Fatalf("write baseline: %v", err)
		}
		t.Logf("baseline rewritten: %s (%d rows)", baseline, len(rows))
	case defaultDir:
		compareCaptureBaseline(t, baseline, rows)
	default:
		t.Logf("$%s set: measured without baseline comparison", captureDirEnv)
	}
}

// measureCapture produces one fixture's row: the traced plain Decode (never
// the diagnostic replay) under the per-image budget and payload
// classification against the known payload map. A failure's stage is the
// furthest stage any ATTEMPTED route reached - level, rung and region routes
// included, not an upright-only replay - with the best route's detail in the
// informational note. Runs on a worker goroutine, so fatal conditions report
// through t.Errorf.
func measureCapture(t *testing.T, dir, rel string, known map[int]captureTruth) captureRow {
	row := captureRow{path: rel, class: captureFail}
	img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Errorf("load %s: %v", rel, err)
		row.stage, row.note = "load-error", err.Error()
		return row
	}
	colors, _ := captureColorCount(rel) // validated before the workers start

	type outcome struct {
		data []byte
		tr   *routeTrace
		err  error
	}
	ch := make(chan outcome, 1)
	start := time.Now()
	go func() {
		data, tr, err := decodeTraced(img)
		ch <- outcome{data, tr, err}
	}()
	var out outcome
	select {
	case out = <-ch:
	case <-time.After(capturePerImageTimeout):
		// The decode goroutine is left running (Decode is not cancellable from
		// outside); the budget is generous enough that only a hang trips this.
		row.wall = time.Since(start)
		row.stage, row.note = "timeout", "decode still running past the per-image budget"
		return row
	}
	row.wall = time.Since(start)

	if out.err != nil {
		row.stage = stageNoFinders.String()
		if best, ok := out.tr.best(); ok {
			row.stage = captureStageString(best.stage)
			row.note = captureRouteNote(best, out.tr, known[colors].side)
		}
		return row
	}
	row.stage = stageDecoded.String()
	switch match := matchKnownPayload(out.data, known); {
	case match == colors:
		row.class = captureOK
	case match != 0:
		row.class = captureOtherCode
		row.note = fmt.Sprintf("decoded the %dc neighbour", match)
	default:
		row.class = captureCorrupt
		row.note = fmt.Sprintf("err=nil, %d bytes match no known payload", len(out.data))
	}
	return row
}

// captureStageString maps a route-trace stage onto the baseline's stage
// vocabulary (the degradation harness's pipelineStage strings), so the TSV
// stays comparable across both attribution mechanisms. An aborted attempt
// carries no stage information and maps to the bottom like readNoFinders.
func captureStageString(s readStage) string {
	switch s {
	case readNoSideSize:
		return stageNoSideSize.String()
	case readNoSample:
		return stageNoSample.String()
	case readSampled:
		return stageSampled.String()
	case readDecoded:
		return stageDecoded.String()
	}
	return stageNoFinders.String()
}

// captureRouteNote renders the best attempt's route - pyramid level,
// pre-rotation, region - plus the located grid estimate when the route got
// far enough to have one, and how many of the attempted routes located the
// TRUE grid. The best attempt is the EARLIEST at the furthest stage (route
// order, not closeness to decoding - stages carry no margin), so the
// true-grid count is the honest signal for the geometry-vs-later-stage
// split, not the best attempt's own grid. Informational only, never
// compared.
func captureRouteNote(best routeAttempt, tr *routeTrace, trueSide image.Point) string {
	var b strings.Builder
	fmt.Fprintf(&b, "best route: L%d rot%g", best.level, best.deg)
	if best.roi >= 0 {
		fmt.Fprintf(&b, " roi%d", best.roi)
	}
	if best.side != (image.Point{}) {
		fmt.Fprintf(&b, " grid %dx%d", best.side.X, best.side.Y)
	}
	trueHits := 0
	for _, a := range tr.attempts {
		if a.side == trueSide {
			trueHits++
		}
	}
	fmt.Fprintf(&b, "; true %dx%d on %d/%d routes", trueSide.X, trueSide.Y, trueHits, len(tr.attempts))
	return b.String()
}

// TestCaptureRouteNote pins the note format and the true-grid counting the
// failure analysis reads, without needing any fixture.
func TestCaptureRouteNote(t *testing.T) {
	tr := &routeTrace{}
	tr.attempts = []routeAttempt{
		{level: 0, deg: 0, roi: -1, stage: readSampled, side: image.Pt(65, 65)},
		{level: 1, deg: 45, roi: -1, stage: readSampled, side: image.Pt(61, 61)},
		{level: 1, deg: 120, roi: 0, stage: readNoFinders},
		{level: 2, deg: 45, roi: -1, stage: readSampled, side: image.Pt(61, 61)},
	}
	best, ok := tr.best()
	if !ok || best.level != 0 || best.side != image.Pt(65, 65) {
		t.Fatalf("best() = %+v, %v; want the earliest sampled attempt (L0 65x65)", best, ok)
	}
	got := captureRouteNote(best, tr, image.Pt(61, 61))
	want := "best route: L0 rot0 grid 65x65; true 61x61 on 2/4 routes"
	if got != want {
		t.Fatalf("captureRouteNote = %q, want %q", got, want)
	}
	got = captureRouteNote(tr.attempts[2], tr, image.Pt(61, 61))
	want = "best route: L1 rot120 roi0; true 61x61 on 2/4 routes"
	if got != want {
		t.Fatalf("captureRouteNote = %q, want %q", got, want)
	}
}

// matchKnownPayload returns the colour count whose ground-truth payload data
// equals byte for byte, or 0 when it matches none.
func matchKnownPayload(data []byte, known map[int]captureTruth) int {
	for colors, truth := range known {
		if bytes.Equal(data, truth.payload) {
			return colors
		}
	}
	return 0
}

// listCaptureFixtures walks dir for capture images and returns their paths
// relative to dir, sorted so the table order is fixed regardless of worker
// scheduling. Skips the test cleanly when the tree is absent or empty.
func listCaptureFixtures(t *testing.T, dir string) []string {
	var fixtures []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		switch strings.ToLower(filepath.Ext(d.Name())) {
		case ".webp", ".png", ".jpg", ".jpeg":
			rel, err := filepath.Rel(dir, path)
			if err != nil {
				return err
			}
			fixtures = append(fixtures, filepath.ToSlash(rel))
		}
		return nil
	})
	if os.IsNotExist(err) {
		t.Skipf("capture fixtures not present at %s (the tree is committed by the user, not by tests); set $%s to point elsewhere", dir, captureDirEnv)
	}
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	if len(fixtures) == 0 {
		t.Skipf("no capture fixtures under %s", dir)
	}
	sort.Strings(fixtures)
	return fixtures
}

// captureTruth is one colour count's ground truth: the payload every capture
// of that code renders, and the code's true module side size (from the
// version in the source filename), which the route notes compare sampled
// grids against.
type captureTruth struct {
	payload []byte
	side    image.Point
}

// captureGroundTruth decodes the pixel-exact source fixtures into the known
// truth map keyed by colour count. Every capture in the set renders one of
// these payloads, so byte equality against the map is the ok / other-code /
// corrupt discriminator. A $JABCAPTURE_DIR tree without its own source/
// directory (frames of the same codes) borrows the repository's canonical one.
func captureGroundTruth(t *testing.T, dir string) map[int]captureTruth {
	srcDir := filepath.Join(dir, "source")
	if _, err := os.Stat(srcDir); err != nil {
		srcDir = filepath.Join(testutil.TestdataPath("highcolor_capture"), "source")
	}
	sources, err := filepath.Glob(filepath.Join(srcDir, "*.png"))
	if err != nil {
		t.Fatalf("glob %s: %v", srcDir, err)
	}
	if len(sources) == 0 {
		t.Skipf("no ground-truth source fixtures under %s; payload classes cannot be established", srcDir)
	}
	sort.Strings(sources)
	known := make(map[int]captureTruth, len(sources))
	for _, path := range sources {
		colors, err := captureColorCount(path)
		if err != nil {
			t.Fatalf("source fixture %s: %v", path, err)
		}
		img, err := loadCaptureImage(path)
		if err != nil {
			t.Fatalf("load source %s: %v", path, err)
		}
		data, err := Decode(img)
		if err != nil {
			t.Fatalf("ground-truth decode %s: %v", path, err)
		}
		// The self-documenting header must agree with the filename, or every
		// classification the map feeds would be built on a mislabeled truth.
		if !bytes.Contains(data, fmt.Appendf(nil, "colors=%d ", colors)) {
			t.Fatalf("source payload %s does not self-document colors=%d", path, colors)
		}
		version, err := captureVersion(path)
		if err != nil {
			t.Fatalf("source fixture %s: %v", path, err)
		}
		if !bytes.Contains(data, fmt.Appendf(nil, "version=%02d ", version)) {
			t.Fatalf("source payload %s does not self-document version=%02d", path, version)
		}
		side := spec.VersionToSize(version)
		known[colors] = captureTruth{payload: data, side: image.Pt(side, side)}
	}
	return known
}

// captureVersion parses the "_v<N>_" side-version tag every source fixture
// name carries (e.g. 64c_ecc10_v11_lorem_ms5.png).
func captureVersion(name string) (int, error) {
	m := regexp.MustCompile(`_v(\d+)_`).FindStringSubmatch(filepath.Base(name))
	if m == nil {
		return 0, fmt.Errorf("no _v<N>_ version tag in %q", filepath.Base(name))
	}
	return strconv.Atoi(m[1])
}

// captureColorCount parses the leading "<N>c_" colour-count prefix every
// fixture name carries (e.g. 128c_side_rot45_normal.webp).
func captureColorCount(name string) (int, error) {
	base := filepath.Base(name)
	i := strings.IndexByte(base, 'c')
	if i <= 0 {
		return 0, fmt.Errorf("no <N>c_ colour-count prefix in %q", base)
	}
	colors, err := strconv.Atoi(base[:i])
	if err != nil {
		return 0, fmt.Errorf("no <N>c_ colour-count prefix in %q: %v", base, err)
	}
	return colors, nil
}

// loadCaptureImage reads and decodes one fixture image.
func loadCaptureImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// formatCaptureReport renders the full row table plus per-directory and
// per-class summaries. Wall times are informational; only path, class and
// stage carry into the baseline.
func formatCaptureReport(rows []captureRow) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "%-52s %-10s %-11s %9s  %s\n", "fixture", "class", "stage", "wall", "note")
	classes := make(map[captureClass]int)
	type dirCount struct{ ok, total int }
	dirs := make(map[string]*dirCount)
	var dirOrder []string
	for _, r := range rows {
		fmt.Fprintf(&b, "%-52s %-10s %-11s %8.1fs  %s\n", r.path, r.class, r.stage, r.wall.Seconds(), r.note)
		classes[r.class]++
		top := r.path
		if i := strings.IndexByte(top, '/'); i >= 0 {
			top = top[:i]
		}
		dc := dirs[top]
		if dc == nil {
			dc = &dirCount{}
			dirs[top] = dc
			dirOrder = append(dirOrder, top)
		}
		dc.total++
		if r.class == captureOK {
			dc.ok++
		}
	}
	fmt.Fprintf(&b, "summary: ok=%d other-code=%d corrupt=%d fail=%d\n",
		classes[captureOK], classes[captureOtherCode], classes[captureCorrupt], classes[captureFail])
	sort.Strings(dirOrder)
	for _, top := range dirOrder {
		fmt.Fprintf(&b, "  %s: %d/%d ok\n", top, dirs[top].ok, dirs[top].total)
	}
	return b.String()
}

// writeCaptureBaseline writes the compared identity of every row (path, class,
// stage) as a TSV. The header comments stay shorter than the longest fixture
// path so IDE column alignment does not widen the first column for them.
func writeCaptureBaseline(path string, rows []captureRow) error {
	var b bytes.Buffer
	b.WriteString("# TestCaptureHarness baseline:\n")
	b.WriteString("# fixture<TAB>class<TAB>stage,\n")
	b.WriteString("# one row per capture fixture.\n")
	b.WriteString("# Advance deliberately with\n")
	b.WriteString("# JABCAPTURE_UPDATE=1; wall times\n")
	b.WriteString("# are not part of the baseline.\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "%s\t%s\t%s\n", r.path, r.class, r.stage)
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

// readCaptureBaseline parses a baseline TSV into rows keyed by fixture path.
func readCaptureBaseline(path string) (map[string]captureRow, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	base := make(map[string]captureRow)
	for n, line := range strings.Split(string(raw), "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Split(line, "\t")
		if len(f) != 3 {
			return nil, fmt.Errorf("%s line %d: want 3 tab-separated fields, got %d", path, n+1, len(f))
		}
		base[f[0]] = captureRow{path: f[0], class: captureClass(f[1]), stage: f[2]}
	}
	return base, nil
}

// compareCaptureBaseline diffs the run against the committed baseline row by
// row. Regressions (class or, within the same class, stage moving down) fail
// the test; improvements are logged so the baseline can be advanced
// deliberately. Rows appearing on only one side fail too - the fixture set
// and the baseline must change together.
func compareCaptureBaseline(t *testing.T, path string, rows []captureRow) {
	base, err := readCaptureBaseline(path)
	if err != nil {
		t.Fatalf("read baseline (seed it with %s=1): %v", captureUpdateEnv, err)
	}
	improved := 0
	for _, r := range rows {
		b, ok := base[r.path]
		if !ok {
			t.Errorf("fixture %s has no baseline row; advance the baseline with %s=1", r.path, captureUpdateEnv)
			continue
		}
		delete(base, r.path)
		if r.class == b.class && r.stage == b.stage {
			continue
		}
		worse := captureClassRank(r.class) < captureClassRank(b.class) ||
			(r.class == b.class && captureStageRank(r.stage) < captureStageRank(b.stage))
		if worse {
			t.Errorf("REGRESSION %s: %s/%s -> %s/%s", r.path, b.class, b.stage, r.class, r.stage)
		} else {
			t.Logf("improved %s: %s/%s -> %s/%s", r.path, b.class, b.stage, r.class, r.stage)
			improved++
		}
	}
	for _, b := range base {
		t.Errorf("baseline row %s missing from the run", b.path)
	}
	if improved > 0 {
		t.Logf("%d rows improved over the baseline; advance it deliberately with %s=1", improved, captureUpdateEnv)
	}
}
