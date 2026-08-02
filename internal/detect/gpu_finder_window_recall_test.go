//go:build !js

package detect

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// The measurement the directional wiring is blocked on: what the fused stage
// keeps and drops relative to the candidate generator it replaces.
//
// The fused kernel is not a drop-in for seekPatternAlong. The CPU seek reports
// every along-line signature and confirms much later, in
// processDirectionalFamilyHit, using walks on the *other two* channels along the
// same base direction. The device cross-check is an off-line walk on the *same*
// channel and has no counterpart at the CPU's seek stage, so it can drop a
// candidate whose blue or red branch would have confirmed it. Under the standing
// priority order that is a decoding-capability regression, and it is invisible to
// every synthetic frame.
//
// **The comparison runs in both directions on purpose.** The two generators are
// not nested: the CPU walk folds runs shorter than three samples and skips past
// an accepted window, while the device tests every window, so each reaches
// candidates the other cannot. Only the CPU-only column is a capability risk;
// the device-only column is downstream work. Reporting one without the other is
// how a set difference gets mistaken for a loss.
//
// Positions are compared spatially, not by key. The generators disagree about
// what a window even is, so matching records to hits by identity would report a
// total mismatch on frames where they agree perfectly about where the finders
// are.
func TestGPUFinderWindowsRecallAgainstTheCPUSweep(t *testing.T) {
	if os.Getenv(benchScanImageEnv) == "" {
		t.Skipf("set %s to a capture to measure recall; synthetic frames cannot answer this", benchScanImageEnv)
	}
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	if !finderScanWorkgroupSupported(device.Info().Limits) {
		t.Skip("adapter cannot launch the workgroup the fused window kernels declare")
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)

	bm := core.BitmapFromImage(benchScanImage(t))
	if bm == nil {
		t.Fatal("convert frame to bitmap")
	}
	masks, width, height := benchScanMasks(t)
	// The detector's own sweep spacing, so the line set is the one a real read
	// pays for rather than a chosen number.
	step := max(height/(2*maxSymbolRows*maxModules), 1)

	for _, deg := range scanDirections[1:] {
		name := fmt.Sprintf("%.0f degrees", deg)
		t.Run(name, func(t *testing.T) {
			geom := sweepGeometry(width, height, deg, float64(step))
			hits := cpuSeekCentres(masks[1], deg, step, width, height)
			survivors := cpuChainSurvivors(bm, masks, deg, step)

			packed, planeWords := packBenchScanMasks(finderScanInterleaved, masks, width, height)
			// Sized from the along-line population, which is what the
			// unfiltered mode emits. An overflow here silently truncates the
			// device set and would read as lost candidates.
			capacity := 1 << 21
			records, meta, counts := runFinderWindows(t, device, kernels,
				finderWindowVariant{"scan", finderScanInterleaved, false, (*gpuDecodeKernels).finderWindowsScan},
				width, height, 1<<1, capacity, packed, planeWords, geom, finderScanEmitUnconfirmed)
			if int(counts[0]) > capacity {
				t.Fatalf("record buffer overflowed at %d of %d, so the sets are truncated", counts[0], capacity)
			}

			var confirmed, all []core.PointF
			for _, w := range records {
				p := finderWindowCentre(w, geom)
				all = append(all, p)
				if meta[w].evidence != 0 {
					confirmed = append(confirmed, p)
				}
			}

			// A finder centre is located to within about a module, so that is
			// the radius at which two generators are talking about the same
			// candidate. Taken from the sweep's own module estimate rather than
			// a pixel constant.
			radius := float64(step) * 2
			pct := func(n, of int) float64 { return 100 * float64(n) / math.Max(1, float64(of)) }

			t.Logf("%s: CPU seek %d, CPU chain survivors %d, device windows %d, device confirmed %d",
				name, len(hits), len(survivors), counts[3], len(confirmed))

			// The raw columns, which say how far the two generators overlap at
			// all. Neither side is a subset of the other, so both are reported.
			t.Logf("%s: raw seek hits with no device window %d (%.1f%%), with no confirmed device record %d (%.1f%%), device-only %d",
				name,
				countUnmatched(hits, all, radius), pct(countUnmatched(hits, all, radius), len(hits)),
				countUnmatched(hits, confirmed, radius), pct(countUnmatched(hits, confirmed, radius), len(hits)),
				countUnmatched(confirmed, hits, radius))

			// **This is the column the wiring decision turns on.** Almost every
			// raw seek hit is a coincidence the CPU chain rejects a stage later,
			// so a difference measured against raw hits says nothing about lost
			// finders. A candidate that survived the whole CPU chain and has no
			// confirmed device record near it is a real capability loss.
			lost := countUnmatched(survivors, confirmed, radius)
			t.Logf("%s: CPU chain survivors with no confirmed device record %d of %d (%.1f%%), none even as a window %d",
				name, lost, len(survivors), pct(lost, len(survivors)),
				countUnmatched(survivors, all, radius))

			// Not an assertion on the numbers: what they should be is exactly
			// the open question, and pinning them here would freeze whichever
			// capture happened to be supplied. The gate is that the run
			// produced comparable populations at all.
			if len(hits) == 0 || counts[3] == 0 {
				t.Fatalf("one generator found nothing, so the columns are not comparable: CPU %d, device %d", len(hits), counts[3])
			}
		})
	}
}

// cpuChainSurvivors runs the whole CPU directional pass and reports the finder
// patterns that survived it, which is the population the device stage actually
// has to preserve. Raw seek hits are not: the chain rejects the overwhelming
// majority of them one stage later, so a set difference taken against them
// measures agreement about noise.
func cpuChainSurvivors(bm *core.Bitmap, masks [3]*core.Bitmap, deg float64, step int) []core.PointF {
	d := &PrimaryDetector{BM: bm, Ch: masks, Mode: normalDetect}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.scanDirectionalFamily(newScanDirection(deg), step, &state)
	out := make([]core.PointF, 0, state.total)
	for _, fp := range state.fps[:state.total] {
		out = append(out, fp.Center)
	}
	return out
}

// cpuSeekCentres is sweepDirection's loop reporting positions instead of
// running the per-hit chain, so it is exactly the candidate set the device stage
// would replace.
func cpuSeekCentres(seek *core.Bitmap, deg float64, step, width, height int) []core.PointF {
	dir := newScanDirection(deg)
	perp := dir.perpendicular()
	nx, ny := perp.dx/perp.pxPerSample, perp.dy/perp.pxPerSample
	qLo, qHi := math.Inf(1), math.Inf(-1)
	for _, c := range [4][2]float64{
		{0, 0}, {float64(width - 1), 0}, {0, float64(height - 1)}, {float64(width - 1), float64(height - 1)},
	} {
		q := c[0]*nx + c[1]*ny
		qLo, qHi = math.Min(qLo, q), math.Max(qHi, q)
	}
	var out []core.PointF
	for q := qLo; q <= qHi; q += float64(step) {
		p0 := core.PointF{X: q * nx, Y: q * ny}
		start, count, ok := clipScanLine(width, height, p0, dir)
		if !ok {
			continue
		}
		for count > 0 {
			centre, _, next, hit := seekPatternAlong(seek, dir, p0.X, p0.Y, start, count)
			if !hit {
				break
			}
			count -= next - start
			start = next
			out = append(out, centre)
		}
	}
	return out
}

// finderWindowCentre resolves a record to a frame position: the midpoint of its
// middle run, in samples along its own line, projected back through the sweep
// basis. This is the arithmetic the survivor contract promises the host, so
// exercising it here also checks the contract is usable.
func finderWindowCentre(w finderWindow, geom finderRunsGeometry) core.PointF {
	line := float64(w.key / 3)
	q := float64(geom.qLo) + line*float64(geom.qStep)
	along := float64(w.boundary[2]+w.boundary[3]) / 2
	return core.PointF{
		X: q*float64(geom.nx) + along*float64(geom.dx),
		Y: q*float64(geom.ny) + along*float64(geom.dy),
	}
}

// countUnmatched reports how many of from have no point of to within radius.
// The grid bucket keeps it linear: a 12 MP frame produces candidates by the tens
// of thousands per angle and the quadratic form is minutes per direction.
func countUnmatched(from, to []core.PointF, radius float64) int {
	type cell struct{ x, y int }
	buckets := make(map[cell][]core.PointF, len(to))
	key := func(p core.PointF) cell {
		return cell{int(math.Floor(p.X / radius)), int(math.Floor(p.Y / radius))}
	}
	for _, p := range to {
		c := key(p)
		buckets[c] = append(buckets[c], p)
	}
	unmatched := 0
	for _, p := range from {
		c := key(p)
		found := false
		for dy := -1; dy <= 1 && !found; dy++ {
			for dx := -1; dx <= 1 && !found; dx++ {
				for _, q := range buckets[cell{c.x + dx, c.y + dy}] {
					if math.Hypot(p.X-q.X, p.Y-q.Y) <= radius {
						found = true
						break
					}
				}
			}
		}
		if !found {
			unmatched++
		}
	}
	return unmatched
}
