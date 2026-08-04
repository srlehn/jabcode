package detect

import (
	"cmp"
	"errors"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/srlehn/jabcode/internal/core"
)

// fakeDirScanner stands in for a pass preparer so the route seam can be tested
// without a device. Only scanDirection matters here; the rest of the interface
// is never reached because these tests call sweepDirectionalFamily directly.
type fakeDirScanner struct {
	hits  []finderDirHit
	err   error
	calls int
}

func TestDirectionalDeviceHitsParallelChainMatchesSerial(t *testing.T) {
	const width, height = 192, 160
	channels := chainTestMasks(width, height, 7, true)
	bm := &core.Bitmap{Width: width, Height: height, Channels: 4, Pix: make([]byte, width*height*4)}
	for i := 0; i < width*height; i++ {
		for c := range 3 {
			bm.Pix[i*4+c] = channels[c].Pix[i]
		}
		bm.Pix[i*4+3] = 255
	}
	rawHits := chainTestRowHits(t, channels[1])
	if len(rawHits) == 0 {
		t.Fatal("chain fixture produced no raw hits")
	}
	hits := make([]finderDirHit, directionalFamilyResultBatch*2+17)
	for i := range hits {
		hit := rawHits[i%len(rawHits)]
		hits[i] = finderDirHit{
			centre: core.PointF{X: hit.center(), Y: float64(hit.y)},
			module: hit.moduleSize(),
		}
	}
	slices.SortFunc(hits, func(a, b finderDirHit) int {
		if c := cmp.Compare(a.centre.Y, b.centre.Y); c != 0 {
			return c
		}
		if c := cmp.Compare(a.centre.X, b.centre.X); c != 0 {
			return c
		}
		return cmp.Compare(a.module, b.module)
	})

	type result struct {
		stats FinderFamilyPassStats
		seed  []float64
		fps   []FinderPattern
		weak  []FinderPattern
		total int
		trace DetectorTrace
	}
	run := func(parallel, traced bool) result {
		d := &PrimaryDetector{BM: bm, Ch: channels, Mode: normalDetect}
		if traced {
			d.Trace = &DetectorTrace{}
		}
		d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
		state := newPrimaryFamilyScan()
		dir := newScanDirection(0)
		if parallel {
			d.processDirectionalFamilyHits(dir, hits, &state)
		} else {
			for _, hit := range hits {
				d.processDirectionalFamilyHit(dir, hit.centre, hit.module, &state)
				if state.done {
					break
				}
			}
		}
		return result{
			stats: d.Stats.Passes[0].FinderFamilyPassStats,
			seed:  append([]float64(nil), d.seedModules...),
			fps:   append([]FinderPattern(nil), state.fps[:state.total]...),
			weak:  append([]FinderPattern(nil), state.weak...),
			total: state.total,
			trace: func() DetectorTrace {
				if d.Trace == nil {
					return DetectorTrace{}
				}
				return *d.Trace
			}(),
		}
	}
	want, got := run(false, false), run(true, false)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("parallel chain differs from serial\nparallel: %#v\nserial:   %#v", got, want)
	}
	if want.stats.BranchBlue+want.stats.BranchRed == 0 {
		t.Fatal("comparison fixture exercised no branch, so it cannot pin the chain effects")
	}
	want, got = run(false, true), run(true, true)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("traced batch chain differs from serial\nbatch:  %#v\nserial: %#v", got, want)
	}
	if len(want.trace.Rejections) == 0 {
		t.Fatal("traced comparison fixture recorded no rejection samples")
	}
}

func TestRetryScanDirectionsReachesParallelDeviceChain(t *testing.T) {
	const width, height = 192, 160
	channels := chainTestMasks(width, height, 7, true)
	rawHits := chainTestRowHits(t, channels[1])
	if len(rawHits) == 0 {
		t.Fatal("chain fixture produced no raw hits")
	}
	hits := make([]finderDirHit, 256)
	for i := range hits {
		hit := rawHits[i%len(rawHits)]
		hits[i] = finderDirHit{
			centre: core.PointF{X: hit.center(), Y: float64(hit.y)},
			module: hit.moduleSize(),
		}
	}
	d := &PrimaryDetector{
		BM: &core.Bitmap{
			Width: width, Height: height, Channels: 4,
			Pix: make([]byte, width*height*4),
		},
		Ch: channels, Mode: normalDetect,
		dirScanner: &fakeDirScanner{hits: hits},
	}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	var picks [finderFamilyCount]familyPick
	d.retryScanDirections(true, false, 1, &picks)
	if d.parallelDirectionalBatches == 0 {
		t.Fatal("production directional retry did not reach the parallel device chain")
	}
}

func TestDirectionalParallelChainPreservesDeferredBitmap(t *testing.T) {
	const width, height = 192, 160
	materialized := 0
	d := &PrimaryDetector{
		BM: &core.Bitmap{Width: width, Height: height, Channels: 4},
		Ch: chainTestMasks(width, height, 7, true),
		materializeBitmap: func() error {
			materialized++
			return nil
		},
	}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.processDirectionalFamilyHits(newScanDirection(45), []finderDirHit{{
		centre: core.PointF{X: -1, Y: -1}, module: 4,
	}}, &state)
	if materialized != 0 {
		t.Fatalf("rejected batch materialized the balanced bitmap %d times", materialized)
	}
	if d.parallelDirectionalBatches == 0 {
		t.Fatal("a deferred bitmap did not enter the parallel chain")
	}
}

func TestDirectionalParallelChainMaterializesBitmapOnce(t *testing.T) {
	const width, height = 192, 160
	channels := chainTestMasks(width, height, 7, true)
	source := &core.Bitmap{
		Width: width, Height: height, Channels: 4,
		Pix: make([]byte, width*height*4),
	}
	for i := 0; i < width*height; i++ {
		for c := range 3 {
			source.Pix[i*4+c] = channels[c].Pix[i]
		}
		source.Pix[i*4+3] = 255
	}
	rawHits := chainTestRowHits(t, channels[1])
	if len(rawHits) == 0 {
		t.Fatal("chain fixture produced no raw hits")
	}
	hits := make([]finderDirHit, 512)
	for i := range hits {
		hit := rawHits[i%len(rawHits)]
		hits[i] = finderDirHit{
			centre: core.PointF{X: hit.center(), Y: float64(hit.y)},
			module: hit.moduleSize(),
		}
	}
	slices.SortFunc(hits, func(a, b finderDirHit) int {
		if c := cmp.Compare(a.centre.Y, b.centre.Y); c != 0 {
			return c
		}
		if c := cmp.Compare(a.centre.X, b.centre.X); c != 0 {
			return c
		}
		return cmp.Compare(a.module, b.module)
	})

	type outcome struct {
		stats FinderFamilyPassStats
		seed  []float64
		fps   []FinderPattern
		weak  []FinderPattern
		total int
	}
	run := func(deferred bool) (outcome, int64) {
		bm := source
		var calls atomic.Int64
		if deferred {
			bm = &core.Bitmap{Width: width, Height: height, Channels: 4}
		}
		d := &PrimaryDetector{BM: bm, Ch: channels, Mode: normalDetect}
		if deferred {
			d.materializeBitmap = func() error {
				calls.Add(1)
				bm.Pix = source.Pix
				return nil
			}
		}
		d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
		state := newPrimaryFamilyScan()
		if deferred {
			d.processDirectionalFamilyHits(newScanDirection(0), hits, &state)
		} else {
			for _, hit := range hits {
				d.processDirectionalFamilyHit(newScanDirection(0), hit.centre, hit.module, &state)
				if state.done {
					break
				}
			}
		}
		return outcome{
			stats: d.Stats.Passes[0].FinderFamilyPassStats,
			seed:  append([]float64(nil), d.seedModules...),
			fps:   append([]FinderPattern(nil), state.fps[:state.total]...),
			weak:  append([]FinderPattern(nil), state.weak...),
			total: state.total,
		}, calls.Load()
	}
	want, _ := run(false)
	got, calls := run(true)
	if calls != 1 {
		t.Fatalf("parallel chain materialized the balanced bitmap %d times, want 1", calls)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("lazy parallel chain differs from serial\nparallel: %#v\nserial:   %#v", got, want)
	}
}

func TestDirectionalParallelBitmapFailureIsReported(t *testing.T) {
	const width, height = 192, 160
	channels := chainTestMasks(width, height, 7, true)
	rawHits := chainTestRowHits(t, channels[1])
	if len(rawHits) == 0 {
		t.Fatal("chain fixture produced no raw hits")
	}
	hits := make([]finderDirHit, 512)
	for i := range hits {
		hit := rawHits[i%len(rawHits)]
		hits[i] = finderDirHit{
			centre: core.PointF{X: hit.center(), Y: float64(hit.y)},
			module: hit.moduleSize(),
		}
	}
	want := errors.New("materialize failed")
	var calls atomic.Int64
	d := &PrimaryDetector{
		BM: &core.Bitmap{Width: width, Height: height, Channels: 4},
		Ch: channels, Mode: normalDetect,
		dirScanner: &fakeDirScanner{},
		materializeBitmap: func() error {
			calls.Add(1)
			return want
		},
	}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.processDirectionalFamilyHits(newScanDirection(0), hits, &state)
	if calls.Load() != 1 {
		t.Fatalf("failed bitmap materialization ran %d times, want 1", calls.Load())
	}
	if !errors.Is(d.materializeErr, want) {
		t.Fatalf("materialize error = %v, want %v", d.materializeErr, want)
	}
	if !errors.Is(d.DirectionalScanError(), want) {
		t.Fatalf("directional scan error = %v, want %v", d.DirectionalScanError(), want)
	}
	if d.dirScanner != nil {
		t.Fatal("bitmap failure did not retire the device route")
	}
	if d.pass().RawHits != 0 {
		t.Fatalf("failed batch replayed %d incomplete hits", d.pass().RawHits)
	}
}

func TestDirectionalParallelResultBufferIsBounded(t *testing.T) {
	const maxBytes = 512 << 10
	bytes := uintptr(directionalFamilyResultBatch) * unsafe.Sizeof(directionalFamilyHitResult{})
	if bytes > maxBytes {
		t.Fatalf("directional replay batch uses %d bytes, limit %d", bytes, maxBytes)
	}
}

func TestDirectionalParallelWorkersObserveCancellation(t *testing.T) {
	const width, height = 192, 160
	channels := chainTestMasks(width, height, 7, true)
	rawHits := chainTestRowHits(t, channels[1])
	if len(rawHits) == 0 {
		t.Fatal("chain fixture produced no raw hits")
	}
	hits := make([]finderDirHit, 256)
	for i := range hits {
		hit := rawHits[i%len(rawHits)]
		hits[i] = finderDirHit{
			centre: core.PointF{X: hit.center(), Y: float64(hit.y)},
			module: hit.moduleSize(),
		}
	}
	var polls atomic.Int64
	d := &PrimaryDetector{
		BM: &core.Bitmap{
			Width: width, Height: height, Channels: 4,
			Pix: make([]byte, width*height*4),
		},
		Ch: channels, Mode: normalDetect,
		Quit: func() bool { return polls.Add(1) > 1 },
	}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.processDirectionalFamilyHits(newScanDirection(45), hits, &state)
	if polls.Load() <= 2 {
		t.Fatalf("parallel workers did not poll cancellation: %d total polls", polls.Load())
	}
	if d.pass().RawHits != 0 {
		t.Fatalf("cancelled parallel chain replayed %d hits", d.pass().RawHits)
	}
}

func (*fakeDirScanner) averagePixelValue([]FinderPattern) ([3]float32, error) {
	return [3]float32{}, nil
}

func (*fakeDirScanner) estimatePitch() (int, int, error) { return 0, 0, nil }

func (*fakeDirScanner) prepare(int, int, []float32, bool, uint32) (
	*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, func() error, error,
) {
	return nil, [3]*core.Bitmap{}, nil, nil, nil
}

func (f *fakeDirScanner) scanDirection(scanDirection, int, int) ([]finderDirHit, error) {
	f.calls++
	return f.hits, f.err
}

// countingDirScanner is a real CPU preparer that counts directional sweeps and
// answers each one with nothing, so the ladder behaves exactly as it does
// without a device while the seam stays observable. It also records which pass
// the first sweep came from, since each pass appends one FinderPassStats.
type countingDirScanner struct {
	finderPassPreparer
	det         *PrimaryDetector
	err         error
	calls       int
	firstAtPass int
}

func (c *countingDirScanner) scanDirection(scanDirection, int, int) ([]finderDirHit, error) {
	c.calls++
	if c.calls == 1 {
		c.firstAtPass = len(c.det.Stats.Passes)
	}
	return nil, c.err
}

// The route seam is only worth anything if the production locate reaches it.
// The seam tests below install dirScanner by hand and call
// sweepDirectionalFamily directly, so all of them stay green if
// retryScanDirections goes back to walking the frame itself or if
// locateFinderFamilies stops handing its preparer over. This drives the whole
// ladder instead and asks only whether the preparer was consulted, and from
// where: every pass runs the directional retry when its row walk does not
// settle, so a rotated capture the raw pass resolves never reaches a later one.
func TestLocateFinderFamiliesReachesTheDirectionalSeam(t *testing.T) {
	r := directionalTestSymbol(t)
	img, _ := renderRotatedRGBA(r, 6, 45)
	bm := core.BitmapFromImage(img)
	BalanceRGB(bm)
	d := &PrimaryDetector{BM: bm, Ch: BinarizerRGB(bm, nil), Mode: IntensiveDetect}
	scanner := &countingDirScanner{finderPassPreparer: cpuFinderPassPreparer{bm: bm}, det: d}
	if _, err := d.locateFinderFamilies(FinderFamilyCurrent.Mask(), scanner); err != nil {
		t.Fatalf("locate returned %v", err)
	}
	if scanner.calls == 0 {
		t.Fatal("the locate ladder never consulted the preparer's directional sweep")
	}
	if scanner.firstAtPass != 1 {
		t.Fatalf("the first device sweep came from pass %d, want the initial pass", scanner.firstAtPass)
	}
}

// The sweep's own fallback keeps a failed device from costing a decode, which
// is exactly what makes a broken kernel invisible: the locate still succeeds
// and only the clock says anything is wrong. The failure has to leave the
// locate, because that is what the read path acts on - it treats a locate error
// as a device it cannot use and redoes the read on the CPU route.
func TestLocateFinderFamiliesReportsADeviceFailure(t *testing.T) {
	want := errors.New("dispatch failed")
	r := directionalTestSymbol(t)
	img, _ := renderRotatedRGBA(r, 6, 45)
	bm := core.BitmapFromImage(img)
	BalanceRGB(bm)
	d := &PrimaryDetector{BM: bm, Ch: BinarizerRGB(bm, nil), Mode: IntensiveDetect}
	scanner := &countingDirScanner{finderPassPreparer: cpuFinderPassPreparer{bm: bm}, det: d, err: want}
	found, err := d.locateFinderFamilies(FinderFamilyCurrent.Mask(), scanner)
	if !errors.Is(err, want) {
		t.Fatalf("locate returned %v, want %v", err, want)
	}
	if found != 0 {
		t.Fatalf("a failed locate published families %v", found)
	}
	// A later locate on the same detector must not inherit it.
	scanner.err = nil
	if _, err := d.locateFinderFamilies(FinderFamilyCurrent.Mask(), scanner); err != nil {
		t.Fatalf("the next locate inherited %v", err)
	}
}

// newRouteDetector is the least a detector needs for sweepDirectionalFamily:
// somewhere to record pass stats, which processDirectionalFamilyHit writes to.
func newRouteDetector() *PrimaryDetector {
	d := &PrimaryDetector{Mode: normalDetect}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	return d
}

// The three route outcomes, which the outcome suites cannot see: they would all
// still pass if the device were never consulted at all.
func TestDirectionalRouteSeam(t *testing.T) {
	dir := newScanDirection(45)
	hit := func(x, y float64) finderDirHit {
		return finderDirHit{centre: core.PointF{X: x, Y: y}, module: 4}
	}

	t.Run("device hits replace the walk", func(t *testing.T) {
		scanner := &fakeDirScanner{hits: []finderDirHit{hit(10, 10), hit(20, 20)}}
		d := newRouteDetector()
		d.dirScanner = scanner
		state := newPrimaryFamilyScan()
		walked := 0
		var seen []core.PointF
		d.sweepDirectionalFamily(dir, 4, 1, &state,
			func(_ scanDirection, c core.PointF, _ float64, _ *primaryFamilyScan) {
				seen = append(seen, c)
			},
			nil,
			func(scanDirection, int, *primaryFamilyScan) { walked++ },
		)
		if scanner.calls != 1 {
			t.Fatalf("the device was consulted %d times, want 1", scanner.calls)
		}
		if walked != 0 {
			t.Fatalf("the CPU walk ran %d times alongside a device sweep", walked)
		}
		if len(seen) != 2 {
			t.Fatalf("the chain saw %d hits, want 2", len(seen))
		}
	})

	t.Run("no device hits fall through to the walk", func(t *testing.T) {
		scanner := &fakeDirScanner{}
		d := newRouteDetector()
		d.dirScanner = scanner
		state := newPrimaryFamilyScan()
		walked := 0
		d.sweepDirectionalFamily(dir, 4, 1, &state,
			func(scanDirection, core.PointF, float64, *primaryFamilyScan) {
				t.Fatal("the chain ran on a device sweep that found nothing")
			},
			nil,
			func(scanDirection, int, *primaryFamilyScan) { walked++ },
		)
		if walked != 1 {
			t.Fatalf("the CPU walk ran %d times, want 1", walked)
		}
		if d.DirectionalScanError() != nil {
			t.Fatalf("an empty sweep was reported as a failure: %v", d.DirectionalScanError())
		}
	})

	// A device error must not read as a device that is merely absent. It is kept
	// for a gate to find, and the route is retired so the rest of the locate does
	// not re-attempt a device that has already failed once.
	t.Run("a device error retires the route and is reported", func(t *testing.T) {
		want := errors.New("dispatch failed")
		scanner := &fakeDirScanner{err: want, hits: []finderDirHit{hit(10, 10)}}
		d := newRouteDetector()
		d.dirScanner = scanner
		state := newPrimaryFamilyScan()
		walked := 0
		walk := func(scanDirection, int, *primaryFamilyScan) { walked++ }
		onHit := func(scanDirection, core.PointF, float64, *primaryFamilyScan) {
			t.Fatal("the chain ran on hits returned alongside an error")
		}
		d.sweepDirectionalFamily(dir, 4, 1, &state, onHit, nil, walk)
		if !errors.Is(d.DirectionalScanError(), want) {
			t.Fatalf("DirectionalScanError = %v, want %v", d.DirectionalScanError(), want)
		}
		if walked != 1 {
			t.Fatalf("the CPU walk ran %d times after a device error, want 1", walked)
		}
		// The second direction must not consult the failed device again.
		d.sweepDirectionalFamily(dir, 4, 1, &state, onHit, nil, walk)
		if scanner.calls != 1 {
			t.Fatalf("a retired device was consulted %d times, want 1", scanner.calls)
		}
		if walked != 2 {
			t.Fatalf("the CPU walk ran %d times over two directions, want 2", walked)
		}
	})
}

// Device blocks reserve output ranges through a global atomic whose ordering is
// unspecified, so the record order differs run to run. That reaches real
// decisions: saveFinderPattern merges and averages centres in arrival order, and
// the scan stops at maxFinderPatterns. The route sorts by identity, so the same
// records in any order reach the chain in the same order.
func TestDirectionalRouteOrdersDeviceHits(t *testing.T) {
	dir := newScanDirection(45)
	hits := []finderDirHit{
		{centre: core.PointF{X: 30, Y: 10}, module: 4},
		{centre: core.PointF{X: 10, Y: 30}, module: 4},
		{centre: core.PointF{X: 10, Y: 10}, module: 5},
		{centre: core.PointF{X: 10, Y: 10}, module: 4},
	}
	run := func(order []finderDirHit) []finderDirHit {
		d := newRouteDetector()
		d.dirScanner = &fakeDirScanner{hits: append([]finderDirHit(nil), order...)}
		state := newPrimaryFamilyScan()
		var seen []finderDirHit
		d.sweepDirectionalFamily(dir, 4, 1, &state,
			func(_ scanDirection, c core.PointF, m float64, _ *primaryFamilyScan) {
				seen = append(seen, finderDirHit{centre: c, module: m})
			},
			nil,
			func(scanDirection, int, *primaryFamilyScan) {
				t.Fatal("the CPU walk ran alongside a device sweep")
			},
		)
		return seen
	}
	forward := run(hits)
	reversed := make([]finderDirHit, len(hits))
	for i, h := range hits {
		reversed[len(hits)-1-i] = h
	}
	if got := run(reversed); len(got) != len(forward) {
		t.Fatalf("reversed input produced %d hits, forward produced %d", len(got), len(forward))
	} else {
		for i := range got {
			if got[i] != forward[i] {
				t.Fatalf("hit %d is %+v from one order and %+v from the other", i, got[i], forward[i])
			}
		}
	}
	if len(forward) != len(hits) {
		t.Fatalf("the chain saw %d of %d hits", len(forward), len(hits))
	}
	for i := 1; i < len(forward); i++ {
		a, b := forward[i-1], forward[i]
		if a.centre.Y > b.centre.Y ||
			(a.centre.Y == b.centre.Y && a.centre.X > b.centre.X) ||
			(a.centre == b.centre && a.module > b.module) {
			t.Fatalf("hits reached the chain unsorted at %d: %+v then %+v", i, a, b)
		}
	}
}
