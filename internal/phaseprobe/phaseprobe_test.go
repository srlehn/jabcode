package phaseprobe

import (
	"bytes"
	"strings"
	"testing"
)

func TestDisabledProbeProducesNoOutput(t *testing.T) {
	active.Store(nil)
	t.Cleanup(func() { active.Store(nil) })
	Mark("ignored")
	Markf("ignored.detail", "value=%d", 1)
	var out bytes.Buffer
	Dump(&out)
	if out.Len() != 0 {
		t.Fatalf("disabled probe wrote %q", out.String())
	}
}

func TestEnabledProbeDumpsCoarseEvents(t *testing.T) {
	active.Store(nil)
	t.Cleanup(func() { active.Store(nil) })
	Enable()
	Mark("first")
	Markf("second", "value=%d", 2)
	var out bytes.Buffer
	Dump(&out)
	text := out.String()
	if !strings.Contains(text, "\tfirst\t\n") {
		t.Fatalf("timing output omits first event:\n%s", text)
	}
	if !strings.Contains(text, "\tsecond\tvalue=2\n") {
		t.Fatalf("timing output omits detailed event:\n%s", text)
	}
	if strings.Index(text, "\tfirst\t") > strings.Index(text, "\tsecond\t") {
		t.Fatalf("timing output changed event order:\n%s", text)
	}
}
