package detect

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// unpackGPUBinarizerMasksScalar is the superseded byte-at-a-time expansion,
// kept purely as the comparison target for the table-driven form.
func unpackGPUBinarizerMasksScalar(bm *core.Bitmap, packedMasks []byte) [3]*core.Bitmap {
	pixelCount := bm.Width * bm.Height
	var rgb [3]*core.Bitmap
	for channel := range rgb {
		rgb[channel] = newBinary(bm)
	}
	pixel := 0
	for word := 0; word < (pixelCount+7)/8; word++ {
		packed := binary.LittleEndian.Uint32(packedMasks[word*4:])
		for lane := 0; lane < 8 && pixel < pixelCount; lane++ {
			mask := packed & 7
			rgb[0].Pix[pixel] = b2byte(mask&1 != 0)
			rgb[1].Pix[pixel] = b2byte(mask&2 != 0)
			rgb[2].Pix[pixel] = b2byte(mask&4 != 0)
			packed >>= 3
			pixel++
		}
	}
	return rgb
}

// TestUnpackGPUBinarizerMasksMatchesScalar pins the table-driven mask expansion
// against the byte-at-a-time form it replaced. The sizes deliberately include
// frames whose pixel count is not a multiple of the eight pixels one packed
// word carries, which is the only place the two forms take different paths.
func TestUnpackGPUBinarizerMasksMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, size := range []struct{ w, h int }{
		{8, 1}, {7, 1}, {1, 1}, {3, 3}, {17, 5}, {64, 64}, {129, 37},
	} {
		bm := core.NewBitmap(size.w, size.h, 4)
		words := (size.w*size.h + 7) / 8
		packed := make([]byte, words*4)
		for i := range packed {
			packed[i] = byte(rng.UintN(256))
		}
		want := unpackGPUBinarizerMasksScalar(bm, packed)
		got := unpackGPUBinarizerMasks(bm, packed)
		for channel := range got {
			if string(got[channel].Pix) != string(want[channel].Pix) {
				t.Fatalf("%dx%d channel %d: table expansion differs from scalar",
					size.w, size.h, channel)
			}
		}
	}
}

// TestUnpackGPUBinarizerMasksSaturates checks the two extremes directly, since
// all-clear and all-set groups are the entries real masks hit most.
func TestUnpackGPUBinarizerMasksSaturates(t *testing.T) {
	bm := core.NewBitmap(16, 2, 4)
	for _, tc := range []struct {
		name string
		fill byte
		want byte
	}{
		{"all clear", 0x00, 0},
		{"all set", 0xFF, 255},
	} {
		packed := make([]byte, (16*2+7)/8*4)
		for i := range packed {
			packed[i] = tc.fill
		}
		got := unpackGPUBinarizerMasks(bm, packed)
		for channel := range got {
			for i, b := range got[channel].Pix {
				if b != tc.want {
					t.Fatalf("%s channel %d pixel %d: got %d, want %d", tc.name, channel, i, b, tc.want)
				}
			}
		}
	}
}
