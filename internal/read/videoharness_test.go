//go:build jabharness

package read

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/srlehn/jabcode/internal/testutil"
)

// TestVideoStreamHarness measures fresh-vs-stream decoding over real video
// frame sequences. It walks $JABSTREAM_DIR (skipping cleanly when unset or
// absent): each <camera>/<fixture-name>/ subdirectory holds one sequence of
// numbered frame images extracted from the private capture videos (e.g.
// `ffmpeg -i name.mov dir/f%03d.png`); the fixture name carries the colour
// count that selects the expected payload, decoded from the committed source
// fixtures at run time. Per sequence it decodes every frame twice - fresh
// (plain Decode, the honest best-of-N baseline) and via one Stream carried
// across the frames - and reports success counts, first-lock frame, payload
// mismatches and per-frame latency.
//
// Cost knobs: $JABSTREAM_MATCH selects sequences by substring of
// <camera>/<fixture>; $JABSTREAM_SKIP (default 0) drops the first N frames;
// $JABSTREAM_STRIDE (default 1) keeps every Nth remaining frame;
// $JABSTREAM_MAXFRAMES caps the frames per sequence after that and
// DEFAULTS TO 6 - an unrestricted pass over all sequences once took an
// hour of all-core time, so unlimited frames (value 0) is an explicit
// opt-in, not a default. Selection mistakes fail loudly instead of
// passing green: a set but unreadable root, a match that selects nothing,
// and a skip that consumes a whole sequence are errors. The source clips
// are Live Photos - ~1.5 s of pre-shutter aiming, then the still moment at
// the embedded still-image-time marker (~60% in, e.g. 1.468 s / frame ~43
// of 72; read it with ffprobe from the mebx track's single-sample
// start_time; it marks the button press, nothing more - decodability can
// scatter anywhere in the clip) - so CONSECUTIVE frames starting at the
// anchor are the default smoke window: e.g. MATCH=<sequence> SKIP=40
// MAXFRAMES=6. Results are measured, never baseline-compared: the frames
// are private inputs. A per-frame watchdog bounds one stuck decode (the
// tripped sequence is abandoned; the unfinished decode leaks its
// goroutine, as in the capture harness). Latency caveat: the stream decode
// of a frame runs SECOND in the same process, on caches (LDPC matrices,
// page cache) the fresh decode just warmed, so the fresh-vs-stream ms
// columns are indicative, not a controlled comparison; the ok counts and
// lock frames are the trustworthy columns, and any latency GATE needs
// separate single-column passes or alternating order instead.
func TestVideoStreamHarness(t *testing.T) {
	root := os.Getenv("JABSTREAM_DIR")
	if root == "" {
		t.Skip("JABSTREAM_DIR not set; skipping real-video stream harness")
	}
	if _, err := os.Stat(root); err != nil {
		// The variable was set deliberately; a broken path must not pass green.
		t.Fatalf("JABSTREAM_DIR: %v", err)
	}
	stride := 1
	if s := os.Getenv("JABSTREAM_STRIDE"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 1 {
			t.Fatalf("JABSTREAM_STRIDE %q: need a positive integer", s)
		}
		stride = v
	}
	maxFrames := 6 // safe default; 0 (unlimited) must be asked for explicitly
	if s := os.Getenv("JABSTREAM_MAXFRAMES"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 0 {
			t.Fatalf("JABSTREAM_MAXFRAMES %q: need a non-negative integer", s)
		}
		maxFrames = v
	}
	skip := 0
	if s := os.Getenv("JABSTREAM_SKIP"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 0 {
			t.Fatalf("JABSTREAM_SKIP %q: need a non-negative integer", s)
		}
		skip = v
	}
	match := os.Getenv("JABSTREAM_MATCH")
	known := captureGroundTruth(t, testutil.TestdataPath("highcolor_capture"))

	var sequences []string
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, cam := range entries {
		if !cam.IsDir() {
			continue
		}
		subs, err := os.ReadDir(filepath.Join(root, cam.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for _, sub := range subs {
			if sub.IsDir() {
				sequences = append(sequences, cam.Name()+"/"+sub.Name())
			}
		}
	}
	slices.Sort(sequences)
	if len(sequences) == 0 {
		t.Skipf("no sequences under %s", root)
	}
	if match != "" {
		kept := sequences[:0]
		for _, seq := range sequences {
			if strings.Contains(seq, match) {
				kept = append(kept, seq)
			}
		}
		if len(kept) == 0 {
			t.Fatalf("JABSTREAM_MATCH %q selects none of the %d sequences", match, len(sequences))
		}
		sequences = kept
	}

	type row struct {
		seq                  string
		frames               int
		freshOK, streamOK    int
		freshBad, streamBad  int // err=nil with wrong payload
		freshLock, streamLok int // first successful frame (1-based, 0 = never)
		freshMS, streamMS    float64
	}
	var rows []row
	for _, seq := range sequences {
		colors, err := captureColorCount(seq)
		if err != nil {
			t.Errorf("%s: %v", seq, err)
			continue
		}
		truth, ok := known[colors]
		if !ok {
			t.Errorf("%s: no ground truth for %dc", seq, colors)
			continue
		}
		dir := filepath.Join(root, filepath.FromSlash(seq))
		var frames []string
		files, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			switch strings.ToLower(filepath.Ext(f.Name())) {
			case ".png", ".jpg", ".jpeg", ".webp":
				frames = append(frames, filepath.Join(dir, f.Name()))
			}
		}
		slices.Sort(frames)
		total := len(frames)
		if skip < len(frames) {
			frames = frames[skip:]
		} else {
			frames = nil
		}
		if stride > 1 {
			kept := frames[:0]
			for i := 0; i < len(frames); i += stride {
				kept = append(kept, frames[i])
			}
			frames = kept
		}
		if maxFrames > 0 && len(frames) > maxFrames {
			frames = frames[:maxFrames]
		}
		if len(frames) == 0 {
			t.Errorf("%s: selection kept 0 of %d frames (skip %d, stride %d)", seq, total, skip, stride)
			continue
		}
		r := row{seq: seq, frames: len(frames)}
		stream := &Stream{}
		verdict := func(data []byte, err error) string {
			switch {
			case err != nil:
				return "fail"
			case string(data) == string(truth.payload):
				return "ok"
			}
			return "BAD"
		}
		// Per-frame watchdog: one pathological frame must not consume the
		// package timeout. Decode is not cancellable, so a tripped budget
		// leaks its goroutine (same documented trade-off as the capture
		// harness) and abandons the sequence - the leaked decode would
		// pollute every later timing.
		const frameBudget = 2 * time.Minute
		budgeted := func(decode func() ([]byte, error)) (data []byte, err error, tripped bool) {
			type result struct {
				data []byte
				err  error
			}
			done := make(chan result, 1)
			go func() {
				data, err := decode()
				done <- result{data, err}
			}()
			select {
			case res := <-done:
				return res.data, res.err, false
			case <-time.After(frameBudget):
				return nil, nil, true
			}
		}
		for i, path := range frames {
			img, err := loadCaptureImage(path)
			if err != nil {
				t.Fatalf("load %s: %v", path, err)
			}
			start := time.Now()
			data, err, tripped := budgeted(func() ([]byte, error) { return Decode(img) })
			if tripped {
				t.Errorf("%s %s: fresh decode exceeded %v; abandoning sequence", seq, filepath.Base(path), frameBudget)
				break
			}
			freshMS := float64(time.Since(start).Microseconds()) / 1000
			r.freshMS += freshMS
			fresh := verdict(data, err)
			if err == nil {
				if fresh == "ok" {
					r.freshOK++
					if r.freshLock == 0 {
						r.freshLock = i + 1
					}
				} else {
					r.freshBad++
				}
			}
			start = time.Now()
			data, err, tripped = budgeted(func() ([]byte, error) { return stream.Decode(img) })
			if tripped {
				t.Errorf("%s %s: stream decode exceeded %v; abandoning sequence", seq, filepath.Base(path), frameBudget)
				break
			}
			streamMS := float64(time.Since(start).Microseconds()) / 1000
			r.streamMS += streamMS
			strm := verdict(data, err)
			if err == nil {
				if strm == "ok" {
					r.streamOK++
					if r.streamLok == 0 {
						r.streamLok = i + 1
					}
				} else {
					r.streamBad++
				}
			}
			// Per-frame progress so a run cut off mid-sequence keeps its evidence.
			t.Logf("  %s %-24s fresh %-4s %8.1f ms  stream %-4s %8.1f ms",
				seq, filepath.Base(path), fresh, freshMS, strm, streamMS)
		}
		rows = append(rows, r)
		t.Logf("%-42s frames %3d  fresh %3d ok (%d bad, lock %d, %6.1f ms/f)  stream %3d ok (%d bad, lock %d, %6.1f ms/f)",
			r.seq, r.frames,
			r.freshOK, r.freshBad, r.freshLock, r.freshMS/float64(r.frames),
			r.streamOK, r.streamBad, r.streamLok, r.streamMS/float64(r.frames))
	}
	var b strings.Builder
	fmt.Fprintf(&b, "\n%-42s %6s %9s %9s %9s %9s %10s %10s\n",
		"sequence", "frames", "fresh-ok", "streamok", "freshbad", "strmbad", "fresh-ms/f", "strm-ms/f")
	for _, r := range rows {
		fmt.Fprintf(&b, "%-42s %6d %9d %9d %9d %9d %10.1f %10.1f\n",
			r.seq, r.frames, r.freshOK, r.streamOK, r.freshBad, r.streamBad,
			r.freshMS/float64(r.frames), r.streamMS/float64(r.frames))
	}
	t.Log(b.String())
}
