//go:build goexperiment.simd && amd64 && !go1.27

package detect

// boxBlurV keeps the scalar sweep on this build. The vertical blur is a
// running sum whose cost is memory bandwidth, not arithmetic, so vector width
// buys nothing: measured on the adverse capture row, the old two-column
// column-major kernel cost 46.5s against the scalar's 21.6s, and rewriting it
// row-major with the lanes across columns still finished at 26.7s. The scalar
// form wins because its loads, stores and divides already stream contiguously.
//
// The two-lane float64 ceiling here is low enough that this is not evidence
// against the wider vectors a go1.27 build can request; it is evidence that
// the access pattern, not the instruction set, decided this kernel.
func boxBlurV(src, dst []float64, w, h, radius int) {
	boxBlurVScalar(src, dst, w, h, radius)
}
