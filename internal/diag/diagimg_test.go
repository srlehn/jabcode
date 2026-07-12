package diag

import "testing"

func TestDiagFilePrefix(t *testing.T) {
	tests := map[string]string{
		"/captures/printed code.PNG": "printed_code",
		"camera.frame.01.jpg":        "camera_frame_01",
		".hidden":                    "hidden",
		"-":                          "stdin",
		"":                           "image",
		"...":                        "image",
	}
	for sourceName, want := range tests {
		if got := diagFilePrefix(sourceName); got != want {
			t.Errorf("diagFilePrefix(%q) = %q, want %q", sourceName, got, want)
		}
	}
}

func TestDiagImageStartSeq(t *testing.T) {
	names := []string{
		"capture_001_balanced.png",
		"capture_099_old.png",
		"capture_1000_previous.png",
		"other_2000_previous.png",
		"001_legacy.png",
		"notes.png",
		"capture_123_nope.jpg",
		"capture_no_sequence.png",
	}
	if got, want := diagImageStartSeq(names, "capture"), 1000; got != want {
		t.Fatalf("diagImageStartSeq = %d, want %d", got, want)
	}
}

func TestDiagImageFilename(t *testing.T) {
	got := diagImageFilename("printed_code", 7, "roi0_", "balanced")
	if want := "printed_code_007_roi0_balanced.png"; got != want {
		t.Fatalf("diagImageFilename = %q, want %q", got, want)
	}
}
