// Package diag is the staged decoder diagnostic behind the jabdiag command:
// it replays the read pipeline stage by stage through the exported hooks of
// the detect and decode packages, writing a human-readable report and,
// optionally, numbered annotated stage images. It only observes; nothing in
// the read path depends on it.
package diag
