package read

import (
	"bytes"
	"image"
	"image/color"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

// complementaryDamageFrames renders two frames whose finder, metadata and
// palette modules are clean but alternate data modules sit just across a
// colour-decision boundary. Each frame loses a different half of the data
// evidence; together every module has one clean observation.
func complementaryDamageFrames(t *testing.T, payload []byte) []image.Image {
	return complementaryDamageFramesForColors(t, payload, 8)
}

func complementaryDamageFramesForColors(t *testing.T, payload []byte, colors int) []image.Image {
	t.Helper()
	const moduleSize = 12
	r, err := encode.Render(encode.Config{Colors: colors, ModuleSize: 1, ECCLevel: 3, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("render complementary symbol: %v", err)
	}
	bm := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, idx := range r.Matrix {
		copy(bm.Pix[i*4:], r.Palette[int(idx)*3:int(idx)*3+3])
		bm.Pix[i*4+3] = 255
	}
	sym := &core.DecodedSymbol{}
	obs, ret := decode.ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observe complementary symbol: %d", ret)
	}
	dataMap := obs.Snapshot().DataMap

	frames := make([]image.Image, 2)
	for damagedParity := range 2 {
		img := image.NewNRGBA(image.Rect(0, 0, r.SideSize.X*moduleSize, r.SideSize.Y*moduleSize))
		for i, idxByte := range r.Matrix {
			x, y := i%r.SideSize.X, i/r.SideSize.X
			idx := int(idxByte)
			rgb := [3]byte{r.Palette[idx*3], r.Palette[idx*3+1], r.Palette[idx*3+2]}
			if dataMap[i] == 0 && i%2 == damagedParity {
				other := idx ^ 1
				for c := range 3 {
					from := float64(rgb[c])
					to := float64(r.Palette[other*3+c])
					rgb[c] = byte(from + (to-from)*0.52 + 0.5)
				}
			}
			for yy := y * moduleSize; yy < (y+1)*moduleSize; yy++ {
				for xx := x * moduleSize; xx < (x+1)*moduleSize; xx++ {
					img.SetNRGBA(xx, yy, color.NRGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255})
				}
			}
		}
		frames[damagedParity] = img
	}
	return frames
}

func TestStreamComplementaryDamagePremise(t *testing.T) {
	payload := bytes.Repeat([]byte("complementary stream evidence "), 3)
	for i, frame := range complementaryDamageFrames(t, payload) {
		if got, err := Decode(frame); err == nil {
			t.Fatalf("frame %d decoded individually as %q; best-of-N must fail", i, got)
		}
	}
}

func TestStreamFusesComplementaryDamage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		colors int
	}{{"4c", 4}, {"8c", 8}} {
		t.Run(tc.name, func(t *testing.T) {
			payload := bytes.Repeat([]byte("complementary stream evidence "), 3)
			frames := complementaryDamageFramesForColors(t, payload, tc.colors)
			for i, frame := range frames {
				if got, err := Decode(frame); err == nil {
					t.Fatalf("frame %d decoded individually as %q", i, got)
				}
			}
			var s Stream
			if got, err := s.Decode(frames[0]); err == nil {
				t.Fatalf("first frame unexpectedly decoded as %q", got)
			}
			got, err := s.Decode(frames[1])
			if err != nil {
				t.Fatalf("complementary aggregate did not decode: %v", err)
			}
			want := isoPayload(payload)
			if !bytes.Equal(got, want) {
				t.Fatalf("complementary aggregate returned %q, want %q", got, want)
			}
			if s.work.correctionChains > 1 {
				t.Fatalf("aggregate exceeded correction quota: %+v", s.work)
			}
		})
	}
}

func TestStreamRejectsMixedComplementaryFrames(t *testing.T) {
	payloadA := bytes.Repeat([]byte("A"), 90)
	payloadB := bytes.Repeat([]byte("B"), 90)
	framesA := complementaryDamageFrames(t, payloadA)
	framesB := complementaryDamageFrames(t, payloadB)
	if _, err := Decode(framesA[0]); err == nil {
		t.Fatal("first mixed-code frame decoded individually")
	}
	if _, err := Decode(framesB[1]); err == nil {
		t.Fatal("second mixed-code frame decoded individually")
	}

	var s Stream
	if got, err := s.Decode(framesA[0]); err == nil {
		t.Fatalf("first mixed-code frame unexpectedly returned %q", got)
	}
	if got, err := s.Decode(framesB[1]); err == nil {
		t.Fatalf("mixed-code evidence returned payload %q", got)
	}
	if s.group.rejects != 1 || len(s.group.snaps) != 1 {
		t.Fatalf("mixed content was not reject-only: rejects=%d snapshots=%d", s.group.rejects, len(s.group.snaps))
	}
}

func TestStreamConfirmedDuplicateSkipsCorrection(t *testing.T) {
	payload := []byte("confirmed duplicate stream frame")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var s Stream
	want := isoPayload(payload)
	if got, err := s.Decode(img); err != nil || !bytes.Equal(got, want) {
		t.Fatalf("first decode = %q, %v", got, err)
	}
	if s.work.correctionChains != 1 {
		t.Fatalf("first frame used %d correction chains, want 1", s.work.correctionChains)
	}
	if got, err := s.Decode(img); err != nil || !bytes.Equal(got, want) {
		t.Fatalf("duplicate decode = %q, %v", got, err)
	}
	if s.work.correctionChains != 0 {
		t.Fatalf("confirmed duplicate used %d correction chains, want 0", s.work.correctionChains)
	}
}

func TestStreamCleanContentChangeDoesNotReturnStalePayload(t *testing.T) {
	payloadA := []byte("stream content version A")
	payloadB := []byte("stream content version B")
	cfg := encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}
	imgA, err := encode.Run(cfg, payloadA)
	if err != nil {
		t.Fatalf("encode A: %v", err)
	}
	imgB, err := encode.Run(cfg, payloadB)
	if err != nil {
		t.Fatalf("encode B: %v", err)
	}
	var s Stream
	wantA := isoPayload(payloadA)
	wantB := isoPayload(payloadB)
	if got, err := s.Decode(imgA); err != nil || !bytes.Equal(got, wantA) {
		t.Fatalf("decode A = %q, %v", got, err)
	}
	if got, err := s.Decode(imgB); err != nil || !bytes.Equal(got, wantB) {
		t.Fatalf("decode B after A = %q, %v; work=%+v rejects=%d snapshots=%d version=%d",
			got, err, s.work, s.group.rejects, len(s.group.snaps), s.group.version)
	}
}

func TestStreamDoesNotAccumulateUnsupportedLayouts(t *testing.T) {
	t.Run("sixteen colours", func(t *testing.T) {
		payload := []byte("sixteen-colour stream stays single-frame")
		img, err := encode.Run(encode.Config{Colors: 16, ModuleSize: 12, ECCLevel: 3, Profile: wire.HighColor, SymbolNumber: 1}, payload)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		s := Stream{profile: wire.HighColor}
		want := isoPayload(payload)
		for i := range 2 {
			got, err := s.Decode(img)
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("frame %d = %q, %v", i, got, err)
			}
			if len(s.group.snaps) != 0 {
				t.Fatalf("frame %d retained unsupported colour evidence", i)
			}
		}
	})

	t.Run("docked secondary", func(t *testing.T) {
		payload := bytes.Repeat([]byte("d"), 100)
		v4 := image.Pt(4, 4)
		img, err := encode.Run(encode.Config{
			Colors: 8, ModuleSize: 12, SymbolNumber: 2,
			SymbolPositions: []int{0, 2}, SymbolVersions: []image.Point{v4, v4}, SymbolECCLevels: []int{0, 0},
		}, payload)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		var s Stream
		want := isoPayload(payload)
		for i := range 2 {
			got, err := s.Decode(img)
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("frame %d = %q, %v", i, got, err)
			}
			if len(s.group.snaps) != 0 || s.work.correctionChains != 1 {
				t.Fatalf("frame %d retained docked evidence or skipped correction: group=%d work=%+v",
					i, len(s.group.snaps), s.work)
			}
		}
	})
}

// TestStreamPrior checks the stream decoder's hypothesis reuse: the first
// rotated frame must record the winning level and rung, and the following
// near-identical frame must decode to the same payload (through the prior
// fast path when it holds, through the fallback search when it does not -
// the payload contract is the same either way).
func TestStreamPrior(t *testing.T) {
	msg := []byte("stream prior round-trip")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var s Stream
	got, err := s.Decode(detect.RotateImage(img, 45))
	if err != nil {
		t.Fatalf("Decode frame 1: %v", err)
	}
	want := isoPayload(msg)
	if string(got) != string(want) {
		t.Fatalf("Decode frame 1: got %q, want %q", got, want)
	}
	if len(s.ring) == 0 {
		t.Fatal("no prior recorded after a successful rotated read")
	}
	if s.ring[0].deg == 0 {
		t.Fatal("rotated read recorded an upright prior")
	}
	got, err = s.Decode(detect.RotateImage(img, 46))
	if err != nil {
		t.Fatalf("Decode frame 2: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("Decode frame 2: got %q, want %q", got, want)
	}
}

// TestStreamQuota asserts the scheduler's per-frame work contract: every
// counter stays within its cap on decodable, rotated and hopeless frames
// alike, retained state stays bounded, and an identical frame sequence
// through a fresh Stream reproduces identical outputs. The exhaustive
// ladder's stages cannot exceed their zero budget by construction - the
// scheduler never calls the region-of-interest proposer or the
// alignment-pattern fallback.
func TestStreamQuota(t *testing.T) {
	msg := []byte("stream quota contract")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	blank := image.NewNRGBA(image.Rect(0, 0, 480, 480))
	frames := []image.Image{
		detect.RotateImage(img, 30),
		blank,
		detect.RotateImage(img, 31),
		img,
		blank,
		detect.RotateImage(img, 29),
	}

	type retainedState struct {
		work                          streamWork
		ring, pending, snaps, rejects int
		version                       uint64
		confirmed                     bool
	}
	run := func() ([][]byte, []error, []retainedState) {
		var s Stream
		want := isoPayload(msg)
		outs := make([][]byte, len(frames))
		errs := make([]error, len(frames))
		states := make([]retainedState, len(frames))
		for i, f := range frames {
			outs[i], errs[i] = s.Decode(f)
			w := s.work
			if w.replayAttempts > 1 || w.uprightScans > 1 || w.rotatedAttempts > 1 || w.correctionChains > 1 {
				t.Errorf("frame %d: work over quota: %+v", i, w)
			}
			if len(s.ring) > streamRingCap || len(s.pending) > streamPendingCap || len(s.group.snaps) > evidenceGroupCap {
				t.Errorf("frame %d: retained state over bounds: ring %d, pending %d, evidence %d",
					i, len(s.ring), len(s.pending), len(s.group.snaps))
			}
			if outs[i] != nil && !bytes.Equal(outs[i], want) {
				t.Errorf("frame %d: wrong payload %q", i, outs[i])
			}
			states[i] = retainedState{
				work: s.work, ring: len(s.ring), pending: len(s.pending), snaps: len(s.group.snaps),
				rejects: s.group.rejects, version: s.group.version, confirmed: s.group.confirmedPayload != nil,
			}
		}
		return outs, errs, states
	}

	outs1, errs1, states1 := run()
	outs2, errs2, states2 := run()
	for i := range frames {
		if !bytes.Equal(outs1[i], outs2[i]) || (errs1[i] == nil) != (errs2[i] == nil) {
			t.Errorf("frame %d: replayed sequence diverged", i)
		}
		if states1[i] != states2[i] {
			t.Errorf("frame %d: retained state diverged: %+v vs %+v", i, states1[i], states2[i])
		}
	}
}
