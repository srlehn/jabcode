package encode

import (
	"strings"
	"testing"
)

func TestISOAnnexDMessageStream(t *testing.T) {
	const message = "JAB Code 2016!"
	wantText := strings.Join([]string{
		"01010", "00001", "00010", "00000", "00011", "11100",
		"01111", "00100", "00101", "11101", "0000", "0011",
		"0001", "0010", "0111", "1101", "0000",
	}, "")
	want := make([]byte, len(wantText))
	for i := range wantText {
		want[i] = wantText[i] - '0'
	}

	seq, encodedLength := analyzeInputData([]byte(message))
	if encodedLength != len(want) {
		t.Fatalf("encoded length = %d, want %d", encodedLength, len(want))
	}
	got, err := encodeData([]byte(message), encodedLength, seq)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("message stream differs\n got %v\nwant %v", got, want)
	}
}
