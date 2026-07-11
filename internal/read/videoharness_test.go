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
// mismatches and per-frame latency. $JABSTREAM_STRIDE (default 1) keeps
// every Nth frame and $JABSTREAM_MAXFRAMES (default 0 = all) caps the frames
// per sequence after striding; failing frames pay a full search twice, so
// unrestricted runs over many sequences need hours (-timeout 240m) - the
// working configuration is STRIDE=6 MAXFRAMES=12, which spans each ~2.4 s
// clip with 12 frames at an effective ~5 fps and finishes in minutes.
// Results are measured, never baseline-compared: the frames are private
// inputs.
func TestVideoStreamHarness(t *testing.T) {
	root := os.Getenv("JABSTREAM_DIR")
	if root == "" {
		t.Skip("JABSTREAM_DIR not set; skipping real-video stream harness")
	}
	if _, err := os.Stat(root); err != nil {
		t.Skipf("JABSTREAM_DIR: %v", err)
	}
	stride := 1
	if s := os.Getenv("JABSTREAM_STRIDE"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 1 {
			t.Fatalf("JABSTREAM_STRIDE %q: need a positive integer", s)
		}
		stride = v
	}
	maxFrames := 0
	if s := os.Getenv("JABSTREAM_MAXFRAMES"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 0 {
			t.Fatalf("JABSTREAM_MAXFRAMES %q: need a non-negative integer", s)
		}
		maxFrames = v
	}
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
			t.Logf("%s: no frames", seq)
			continue
		}
		r := row{seq: seq, frames: len(frames)}
		stream := &Stream{}
		for i, path := range frames {
			img, err := loadCaptureImage(path)
			if err != nil {
				t.Fatalf("load %s: %v", path, err)
			}
			start := time.Now()
			data, err := Decode(img)
			r.freshMS += float64(time.Since(start).Microseconds()) / 1000
			if err == nil {
				if string(data) == string(truth.payload) {
					r.freshOK++
					if r.freshLock == 0 {
						r.freshLock = i + 1
					}
				} else {
					r.freshBad++
				}
			}
			start = time.Now()
			data, err = stream.Decode(img)
			r.streamMS += float64(time.Since(start).Microseconds()) / 1000
			if err == nil {
				if string(data) == string(truth.payload) {
					r.streamOK++
					if r.streamLok == 0 {
						r.streamLok = i + 1
					}
				} else {
					r.streamBad++
				}
			}
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
