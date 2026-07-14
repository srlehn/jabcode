package encode

import "testing"

func TestApplyRule3RectangularMatrix(t *testing.T) {
	const width, height = 29, 25
	matrix := make([]int, width*height)
	got := applyRule3(matrix, width, height)
	want := height*(maskW3+width-5) + width*(maskW3+height-5)
	if got != want {
		t.Fatalf("applyRule3 = %d, want %d", got, want)
	}
}
