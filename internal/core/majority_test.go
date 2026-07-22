package core

import "testing"

func TestMajority5Row(t *testing.T) {
	src := []byte{255, 0, 255, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255}
	got := append([]byte(nil), src...)
	want := append([]byte(nil), src...)
	for j := 2; j < len(src)-2; j++ {
		sum := 0
		for k := -2; k <= 2; k++ {
			if src[j+k] != 0 {
				sum++
			}
		}
		if sum > 2 {
			want[j] = 255
		} else {
			want[j] = 0
		}
	}
	Majority5Row(src, got, len(src))
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestMajority5Columns(t *testing.T) {
	const width, height = 37, 29
	src := make([]byte, width*height)
	for i := range src {
		if (i*17+i/width*7)%11 < 5 {
			src[i] = 255
		}
	}
	got := make([]byte, len(src))
	want := make([]byte, len(src))
	Majority5Columns(src, got, width, height)
	for y := 2; y < height-2; y++ {
		for x := 2; x < width-2; x++ {
			sum := 0
			for yy := y - 2; yy <= y+2; yy++ {
				if src[yy*width+x] != 0 {
					sum++
				}
			}
			if sum > 2 {
				want[y*width+x] = 255
			}
		}
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("column majority byte %d = %d, want %d", i, got[i], want[i])
		}
	}
}
