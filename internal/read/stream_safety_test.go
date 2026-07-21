package read

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

func deterministicNoise(bounds image.Rectangle, seed uint32) *image.NRGBA {
	img := image.NewNRGBA(bounds)
	state := seed
	for i := 0; i < len(img.Pix); i += 4 {
		state = state*1664525 + 1013904223
		img.Pix[i+0] = byte(state >> 24)
		img.Pix[i+1] = byte(state >> 16)
		img.Pix[i+2] = byte(state >> 8)
		img.Pix[i+3] = 255
	}
	return img
}

func finderLikeFrame(bounds image.Rectangle) *image.NRGBA {
	img := image.NewNRGBA(bounds)
	draw.Draw(img, bounds, &image.Uniform{C: color.NRGBA{R: 245, G: 245, B: 245, A: 255}}, image.Point{}, draw.Src)
	colors := []color.NRGBA{
		{A: 255},
		{R: 255, A: 255},
		{G: 255, A: 255},
		{B: 255, A: 255},
	}
	centers := []image.Point{
		bounds.Min.Add(image.Pt(28, 28)),
		image.Pt(bounds.Max.X-29, bounds.Min.Y+28),
		image.Pt(bounds.Max.X-29, bounds.Max.Y-29),
		image.Pt(bounds.Min.X+28, bounds.Max.Y-29),
	}
	for corner, center := range centers {
		for ring := 3; ring >= 0; ring-- {
			radius := 3 + ring*4
			rect := image.Rect(center.X-radius, center.Y-radius, center.X+radius+1, center.Y+radius+1)
			draw.Draw(img, rect.Intersect(bounds), &image.Uniform{C: colors[(corner+ring)%len(colors)]}, image.Point{}, draw.Src)
		}
	}
	return img
}

func offsetFrame(src image.Image, minimum image.Point) *image.NRGBA {
	size := src.Bounds().Size()
	bounds := image.Rectangle{Min: minimum, Max: minimum.Add(size)}
	dst := image.NewNRGBA(bounds)
	draw.Draw(dst, bounds, src, src.Bounds().Min, draw.Src)
	return dst
}

func TestStreamHostileChangingSequenceStaysBounded(t *testing.T) {
	payloadA := bytes.Repeat([]byte{0x00, 0x5c, 0x41, 0xff}, 24)
	payloadB := bytes.Repeat([]byte{0x00, 0x5c, 0x42, 0x80}, 24)
	version := image.Pt(7, 7)
	cfg := encode.Config{
		Colors: 8, ModuleSize: 5, ECCLevel: spec.DefaultECCLevel,
		Format: wire.EncodeISO23634, Opaque: true, SymbolNumber: 1,
		SymbolPositions: []int{0}, SymbolVersions: []image.Point{version},
		SymbolECCLevels: []int{spec.DefaultECCLevel},
	}
	cleanA, err := encode.Run(cfg, payloadA)
	if err != nil {
		t.Fatalf("encode A: %v", err)
	}
	cleanB, err := encode.Run(cfg, payloadB)
	if err != nil {
		t.Fatalf("encode B: %v", err)
	}
	damagedA := complementaryDamageFrames(t, payloadA)
	damagedB := complementaryDamageFrames(t, payloadB)

	frames := []image.Image{
		image.NewNRGBA(image.Rectangle{}),
		image.NewNRGBA(image.Rect(7, -3, 8, -2)),
		deterministicNoise(image.Rect(-11, 5, 246, 198), 1),
		finderLikeFrame(image.Rect(13, -9, 332, 232)),
		offsetFrame(cleanA, image.Pt(-17, 23)),
		damagedA[0],
		deterministicNoise(image.Rect(0, 0, 301, 217), 2),
		damagedB[1],
		offsetFrame(cleanB, image.Pt(31, -15)),
		finderLikeFrame(image.Rect(-5, 19, 252, 212)),
		damagedA[1],
		damagedB[0],
	}
	for round := range 3 {
		frames = append(frames,
			deterministicNoise(image.Rect(-round, round, 257+round*2, 194+round*2), uint32(10+round)),
			offsetFrame(cleanA, image.Pt(round*3-4, 5-round)),
			finderLikeFrame(image.Rect(round-7, 11-round, 250+round, 204+round)),
			offsetFrame(cleanB, image.Pt(9-round, round*2-6)),
		)
	}

	var stream Stream
	seenA, seenB := false, false
	for index, frame := range frames {
		if index == len(frames)/2 {
			stream.Reset()
			if len(stream.ring) != 0 || len(stream.pending) != 0 || len(stream.group.snaps) != 0 {
				t.Fatalf("Reset retained state: ring=%d pending=%d evidence=%d", len(stream.ring), len(stream.pending), len(stream.group.snaps))
			}
		}
		message, err := stream.DecodeMessage(frame)
		if err == nil {
			switch {
			case bytes.Equal(message.Data, payloadA):
				seenA = true
			case bytes.Equal(message.Data, payloadB):
				seenB = true
			default:
				t.Fatalf("frame %d returned transition mixture %x", index, message.Data)
			}
		}
		if stream.work.replayAttempts > 1 || stream.work.uprightScans > 1 ||
			stream.work.rotatedAttempts > 1 || stream.work.enlargedAttempts > 1 ||
			stream.work.correctionChains > 1 {
			t.Fatalf("frame %d exceeded work quota: %+v", index, stream.work)
		}
		if len(stream.ring) > streamRingCap || len(stream.pending) > streamPendingCap ||
			len(stream.group.snaps) > evidenceGroupCap {
			t.Fatalf("frame %d exceeded retained-state caps: ring=%d pending=%d evidence=%d",
				index, len(stream.ring), len(stream.pending), len(stream.group.snaps))
		}
	}
	if !seenA || !seenB {
		t.Fatalf("clean changing sequence decoded A=%t B=%t", seenA, seenB)
	}
}

func TestStreamResetPreservesOnlyForcedCapability(t *testing.T) {
	ordinary := Stream{
		capabilities: wire.PreV2C.Mask(),
		forced:       false,
		ring:         []streamPrior{{side: 100}},
		pending:      []streamHyp{{side: 200}},
		gen:          9,
	}
	ordinary.Reset()
	if ordinary.capabilities != 0 || ordinary.forced || len(ordinary.ring) != 0 ||
		len(ordinary.pending) != 0 || ordinary.gen != 0 {
		t.Fatalf("ordinary Reset = %+v", ordinary)
	}

	restricted := NewStreamOnly(wire.ISO23634)
	restricted.ring = []streamPrior{{side: 100}}
	restricted.pending = []streamHyp{{side: 200}}
	restricted.gen = 9
	restricted.Reset()
	if restricted.capabilities != wire.ISO23634.Mask() || !restricted.forced ||
		len(restricted.ring) != 0 || len(restricted.pending) != 0 || restricted.gen != 0 {
		t.Fatalf("restricted Reset = %+v", restricted)
	}
}
