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
