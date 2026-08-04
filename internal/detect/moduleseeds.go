package detect

// The print retry asks two questions of the module-size estimates a pass
// collects: how many there were, and what their median is. It never reads an
// individual value. A directional device sweep produces hundreds of thousands
// of them per level, so keeping the values would mean carrying every raw hit
// back from the device to answer a two-number question. A bucketed histogram
// answers both, merges associatively, and reduces on the device.

const (
	// moduleSeedsPerPixel is the histogram resolution. The median feeds a blur
	// radius that is itself quantized to whole pixels after a division by four,
	// so quarter-pixel buckets are finer than anything downstream can observe.
	moduleSeedsPerPixel = 4
	// moduleSeedsBuckets covers module sizes up to 256 px, past which a capture
	// has one module per frame and no retry tier applies.
	moduleSeedsBuckets = 1024
)

// moduleSeeds is the per-signature module-size distribution of one detection's
// raw hits. The zero value is an empty accumulator.
type moduleSeeds struct {
	count   int
	buckets [moduleSeedsBuckets]int32
}

// add records one module-size estimate. Sizes at or above the covered range
// land in the last bucket, which keeps the median well defined without
// rejecting a capture whose modules are enormous.
func (s *moduleSeeds) add(moduleSize float64) {
	if moduleSize <= 0 {
		return
	}
	bucket := int(moduleSize * moduleSeedsPerPixel)
	if bucket >= moduleSeedsBuckets {
		bucket = moduleSeedsBuckets - 1
	}
	s.buckets[bucket]++
	s.count++
}

// addBucket records n hits already reduced into one bucket, which is how a
// device histogram merges into the host accumulator.
func (s *moduleSeeds) addBucket(bucket int, n int) {
	if n <= 0 || bucket < 0 || bucket >= moduleSeedsBuckets {
		return
	}
	s.buckets[bucket] += int32(n)
	s.count += n
}

func (s *moduleSeeds) len() int { return s.count }

func (s *moduleSeeds) reset() { *s = moduleSeeds{} }

// median returns the bucket midpoint holding the middle observation, or zero
// for an empty accumulator. The midpoint rather than the lower edge keeps the
// estimate centred in the same way the exact median of a sorted slice was.
func (s *moduleSeeds) median() float64 {
	if s.count == 0 {
		return 0
	}
	target := s.count / 2
	seen := 0
	for bucket, n := range s.buckets {
		seen += int(n)
		if seen > target {
			return (float64(bucket) + 0.5) / moduleSeedsPerPixel
		}
	}
	return (float64(moduleSeedsBuckets-1) + 0.5) / moduleSeedsPerPixel
}
