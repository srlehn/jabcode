// Package detect locates JAB Code symbols in an image: channel balancing and
// binarization (with descreen retries sized from the image's own lattice
// pitch), finder- and alignment-pattern detection, side-size estimation,
// perspective sampling of the module grid, the coarse orientation probe, and
// the region-of-interest proposer. Decoding the sampled matrix lives in the
// sibling decode package; the read package coordinates the two.
package detect
