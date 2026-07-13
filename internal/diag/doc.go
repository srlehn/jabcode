// Package diag renders the observation trace produced by the authoritative
// decoder behind jabcode decode --diag. It writes a human-readable report and,
// optionally, numbered annotated stage images. It never reruns a decode stage;
// nothing in the read path depends on it.
package diag
