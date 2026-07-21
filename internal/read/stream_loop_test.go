package read

import (
	"bytes"
	"image"
	"image/color"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/encode"
)

const loopModuleSize = 12

// loopCode is one distinct code of a repeating changing-code loop, rendered
// once at one pixel per module and reused to paint clean, complementary-damaged
// and transition frames. Every code in a loop shares the encoder config, so
// their footprint, palette, fixed patterns and metadata are identical and only
// the data modules differ: the real product case, where a symbol at one screen
// location cycles through payloads while the camera holds still.
type loopCode struct {
	payload []byte
	side    image.Point
	palette []byte
	matrix  []byte
	dataMap []byte
}

func newLoopCode(t *testing.T, payload []byte) loopCode {
	t.Helper()
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 1, ECCLevel: 3, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("render loop code: %v", err)
	}
	bm := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, idx := range r.Matrix {
		copy(bm.Pix[i*4:], r.Palette[int(idx)*3:int(idx)*3+3])
		bm.Pix[i*4+3] = 255
	}
	sym := &core.DecodedSymbol{}
	obs, ret := decode.ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observe loop code: %d", ret)
	}
	return loopCode{payload: payload, side: r.SideSize, palette: r.Palette, matrix: r.Matrix, dataMap: obs.Snapshot().DataMap}
}

// paint renders one frame of the code. A negative damage yields a clean frame;
// damage 0 or 1 pushes that parity of the data modules just past their colour
// boundary, so the two parities lose complementary halves of the data evidence
// and neither decodes alone (the existing complementary-damage construction).
func (c loopCode) paint(damage int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, c.side.X*loopModuleSize, c.side.Y*loopModuleSize))
	for i, idxByte := range c.matrix {
		x, y := i%c.side.X, i/c.side.X
		idx := int(idxByte)
		rgb := [3]byte{c.palette[idx*3], c.palette[idx*3+1], c.palette[idx*3+2]}
		if damage >= 0 && c.dataMap[i] == 0 && i%2 == damage {
			other := idx ^ 1
			for ch := range 3 {
				from := float64(rgb[ch])
				to := float64(c.palette[other*3+ch])
				rgb[ch] = byte(from + (to-from)*0.52 + 0.5)
			}
		}
		for yy := y * loopModuleSize; yy < (y+1)*loopModuleSize; yy++ {
			for xx := x * loopModuleSize; xx < (x+1)*loopModuleSize; xx++ {
				img.SetNRGBA(xx, yy, color.NRGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255})
			}
		}
	}
	return img
}

// blendFrames is a crossfade transition: every pixel is the midpoint of the two
// codes. Fixed patterns, palette and metadata are identical between the codes,
// so the blend keeps clean structure while landing every differing data module
// on its colour-decision boundary: a frame that locates cleanly but must never
// decode to either code.
func blendFrames(a, b *image.NRGBA) *image.NRGBA {
	out := image.NewNRGBA(a.Rect)
	for i := range a.Pix {
		out.Pix[i] = byte((int(a.Pix[i]) + int(b.Pix[i])) / 2)
	}
	return out
}

// shutterFrames is a rolling-shutter disagreement: the top half of the frame
// captured one code and the bottom half the next. Structure stays clean but the
// two data halves carry different payloads, so it must never decode to either.
func shutterFrames(a, b *image.NRGBA) *image.NRGBA {
	out := image.NewNRGBA(a.Rect)
	copy(out.Pix, a.Pix)
	split := a.Rect.Dy() / 2 * a.Stride
	copy(out.Pix[split:], b.Pix[split:])
	return out
}

// loopFrame is one frame of a generated loop with the ground truth needed to
// gate it: the code it shows (or -1 for a transition), and the loop iteration.
type loopFrame struct {
	img  image.Image
	code int
	loop int
	kind string
}

// repeatingLoopFrames builds a repeating changing-code stream: each code shown
// for dwell clean frames, with a blended and a rolling-shutter transition frame
// between consecutive codes, repeated for the given number of loops so every
// code reappears. All codes must share one footprint.
func repeatingLoopFrames(t *testing.T, codes []loopCode, dwell, loops int) []loopFrame {
	t.Helper()
	for i := 1; i < len(codes); i++ {
		if codes[i].side != codes[0].side {
			t.Fatalf("loop codes must share a footprint: code %d %v != %v", i, codes[i].side, codes[0].side)
		}
	}
	clean := make([]*image.NRGBA, len(codes))
	for i := range codes {
		clean[i] = codes[i].paint(-1)
	}
	// The transition between one ordered code pair is identical on every loop,
	// so cache it: a long loop then allocates a fixed handful of frames rather
	// than one transition per iteration.
	blendCache := map[[2]int]*image.NRGBA{}
	shutterCache := map[[2]int]*image.NRGBA{}
	var seq []loopFrame
	prev := -1
	for l := range loops {
		for ci := range codes {
			if prev >= 0 {
				key := [2]int{prev, ci}
				if blendCache[key] == nil {
					blendCache[key] = blendFrames(clean[prev], clean[ci])
					shutterCache[key] = shutterFrames(clean[prev], clean[ci])
				}
				seq = append(seq, loopFrame{blendCache[key], -1, l, "blend"})
				seq = append(seq, loopFrame{shutterCache[key], -1, l, "shutter"})
			}
			for range dwell {
				seq = append(seq, loopFrame{clean[ci], ci, l, "clean"})
			}
			prev = ci
		}
	}
	return seq
}

// TestStreamRepeatingLoop drives the product use case: a symbol at one location
// cycling through a fixed set of payloads, repeated. It gates the invariants
// that hold regardless of the still-open emission policy (item 3): a transition
// frame never decodes, every emission is the payload actually on screen (never
// a stale, mixed or neighbouring code), a reappearing code decodes again
// without stale fusion, the per-frame work quota and retained-state bounds hold,
// and the whole sequence is deterministic on replay.
func TestStreamRepeatingLoop(t *testing.T) {
	codes := []loopCode{
		newLoopCode(t, bytes.Repeat([]byte("A"), 90)),
		newLoopCode(t, bytes.Repeat([]byte("B"), 90)),
		newLoopCode(t, bytes.Repeat([]byte("C"), 90)),
	}
	want := make([][]byte, len(codes))
	for i := range codes {
		want[i] = isoPayload(codes[i].payload)
	}
	const dwell, loops = 3, 2
	seq := repeatingLoopFrames(t, codes, dwell, loops)

	run := func() [][]byte {
		var s Stream
		outs := make([][]byte, len(seq))
		emitted := map[[2]int]bool{}
		for i, f := range seq {
			data, err := s.Decode(f.img)
			outs[i] = data
			w := s.work
			if w.replayAttempts > 1 || w.uprightScans > 1 || w.rotatedAttempts > 1 ||
				w.enlargedAttempts > 1 || w.correctionChains > 1 {
				t.Fatalf("frame %d (%s) over quota: %+v", i, f.kind, w)
			}
			if len(s.ring) > streamRingCap || len(s.pending) > streamPendingCap ||
				len(s.group.snaps) > evidenceGroupCap {
				t.Fatalf("frame %d (%s) retained state over bounds: ring %d pending %d evidence %d",
					i, f.kind, len(s.ring), len(s.pending), len(s.group.snaps))
			}
			if f.code < 0 {
				if err == nil {
					t.Fatalf("frame %d (%s transition) emitted %q; a transition must not decode", i, f.kind, data)
				}
				continue
			}
			if err == nil {
				if !bytes.Equal(data, want[f.code]) {
					t.Fatalf("frame %d (loop %d code %d) emitted %q, want %q", i, f.loop, f.code, data, want[f.code])
				}
				emitted[[2]int{f.loop, f.code}] = true
			}
		}
		for l := range loops {
			for ci := range codes {
				if !emitted[[2]int{l, ci}] {
					t.Fatalf("code %d never emitted on loop %d", ci, l)
				}
			}
		}
		return outs
	}

	outs1 := run()
	outs2 := run()
	for i := range seq {
		if !bytes.Equal(outs1[i], outs2[i]) {
			t.Fatalf("frame %d: replayed sequence diverged: %q vs %q", i, outs1[i], outs2[i])
		}
	}
}

// TestStreamLongLoopBounded runs the changing-code loop for many iterations and
// proves the boundedness half of the stream contract: no retained store grows
// with sequence length (the search ring, the carried-hypothesis queue and the
// content group's snapshots all stay within their caps on every frame), the
// per-frame work quota holds throughout, no transition ever decodes, and every
// emission is the on-screen payload. It also pins determinism at length: the
// same long sequence through a fresh Stream reproduces byte-identical outputs
// and identical retained-state sizes on every frame, so eviction and evidence
// reduction cannot depend on map, completion or wall-clock order.
func TestStreamLongLoopBounded(t *testing.T) {
	codes := []loopCode{
		newLoopCode(t, bytes.Repeat([]byte("A"), 90)),
		newLoopCode(t, bytes.Repeat([]byte("B"), 90)),
		newLoopCode(t, bytes.Repeat([]byte("C"), 90)),
	}
	want := make([][]byte, len(codes))
	for i := range codes {
		want[i] = isoPayload(codes[i].payload)
	}
	const dwell, loops = 2, 24
	seq := repeatingLoopFrames(t, codes, dwell, loops)

	type state struct{ ring, pending, snaps int }
	run := func() ([][]byte, []state) {
		var s Stream
		outs := make([][]byte, len(seq))
		states := make([]state, len(seq))
		for i, f := range seq {
			data, err := s.Decode(f.img)
			outs[i] = data
			w := s.work
			if w.replayAttempts > 1 || w.uprightScans > 1 || w.rotatedAttempts > 1 ||
				w.enlargedAttempts > 1 || w.correctionChains > 1 {
				t.Fatalf("frame %d over quota: %+v", i, w)
			}
			if len(s.ring) > streamRingCap || len(s.pending) > streamPendingCap ||
				len(s.group.snaps) > evidenceGroupCap {
				t.Fatalf("frame %d retained state over bounds: ring %d pending %d snaps %d",
					i, len(s.ring), len(s.pending), len(s.group.snaps))
			}
			switch {
			case f.code < 0 && err == nil:
				t.Fatalf("frame %d (%s) transition emitted %q", i, f.kind, data)
			case f.code >= 0 && err == nil && !bytes.Equal(data, want[f.code]):
				t.Fatalf("frame %d (code %d) emitted %q, want %q", i, f.code, data, want[f.code])
			}
			states[i] = state{len(s.ring), len(s.pending), len(s.group.snaps)}
		}
		return outs, states
	}

	outs1, states1 := run()
	outs2, states2 := run()
	for i := range seq {
		if !bytes.Equal(outs1[i], outs2[i]) || states1[i] != states2[i] {
			t.Fatalf("frame %d diverged on replay: %q/%+v vs %q/%+v",
				i, outs1[i], states1[i], outs2[i], states2[i])
		}
	}
}
