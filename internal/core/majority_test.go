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

func TestMajority5VerticalRow(t *testing.T) {
	rows := [5][]byte{
		make([]byte, 39), make([]byte, 39), make([]byte, 39), make([]byte, 39), make([]byte, 39),
	}
	for r := range rows {
		for x := range rows[r] {
			if (r*11+x*7)%13 < 6 {
				rows[r][x] = 255
			}
		}
	}
	got := make([]byte, 39)
	want := make([]byte, 39)
	for j := 2; j < 37; j++ {
		sum := 0
		for _, row := range rows {
			if row[j] != 0 {
				sum++
			}
		}
		if sum > 2 {
			want[j] = 255
		}
	}
	Majority5VerticalRow(rows, got, len(got))
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %d, want %d", i, got[i], want[i])
		}
	}
}
