//go:build !js

package detect

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/srlehn/vulki"
)

// finderWindow is one accepted five-run signature: the line and channel it was
// found on, and the six boundaries that define it. It is the record's identity,
// kept free of the metadata so the two can be compared separately: a set of
// windows against the oracle's set, and each window's metadata against what the
// oracle's own walks measured.
type finderWindow struct {
	key      uint32
	boundary [6]uint32
}

// finderRecordMeta is what a record says beyond its identity: which of the three
// walks confirmed it, and the module size that walk measured.
type finderRecordMeta struct {
	evidence uint32
	module   float32
}

// runFinderWindows dispatches the fused prototype and returns the survivors it
// wrote, the true accepted count and the subset whose inner runs are at least
// three samples.
// finderWindowVariant is one fused kernel: the ballot form where the device
// supports it, and the portable workgroup-scan form everywhere. Both are
// checked, because the fallback is only a fallback if it is known to agree.
type finderWindowVariant struct {
	name     string
	layout   finderScanLayout
	subgroup bool
	kernel   func(*gpuDecodeKernels, finderScanLayout) (*vulki.Kernel, error)
}

// finderWindowVariants lists the fused kernels to check. Both declare a 256-lane
// workgroup, which Vulkan Core does not guarantee, so an adapter that cannot
// launch one has no variant to check rather than a failing one.
func finderWindowVariants(t *testing.T, kernels *gpuDecodeKernels) []finderWindowVariant {
	t.Helper()
	if !finderScanWorkgroupSupported(kernels.device.Info().Limits) {
		t.Skip("adapter cannot launch the workgroup the fused window kernels declare")
	}
	var out []finderWindowVariant
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		out = append(out,
			finderWindowVariant{"ballot " + layout.name(), layout, true, (*gpuDecodeKernels).finderWindowsBallot},
			finderWindowVariant{"scan " + layout.name(), layout, false, (*gpuDecodeKernels).finderWindowsScan},
		)
	}
	return out
}

func runFinderWindows(
	t *testing.T,
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	variant finderWindowVariant,
	width, height int,
	channelMask uint32,
	capacity int,
	packed []byte,
	planeWords int,
	geom finderRunsGeometry,
	flags uint32,
) (found []finderWindow, meta map[finderWindow]finderRecordMeta, counts [finderWindowCounterCount]uint32) {
	t.Helper()
	kernel, err := variant.kernel(kernels, variant.layout)
	if err != nil {
		t.Fatalf("compile fused windows %s: %v", variant.name, err)
	}
	masks, err := device.NewBuffer(uint64(len(packed)))
	if err != nil {
		t.Fatalf("allocate masks: %v", err)
	}
	defer func() { _ = masks.Close() }()
	out, err := device.NewBuffer(uint64(capacity * finderWindowRecordWords * 4))
	if err != nil {
		t.Fatalf("allocate survivors: %v", err)
	}
	defer func() { _ = out.Close() }()
	countBuf, err := device.NewBuffer(finderWindowCounterCount * 4)
	if err != nil {
		t.Fatalf("allocate counters: %v", err)
	}
	defer func() { _ = countBuf.Close() }()
	paramBuf, err := device.NewBuffer(finderScanParamsBytes)
	if err != nil {
		t.Fatalf("allocate params: %v", err)
	}
	defer func() { _ = paramBuf.Close() }()

	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, masks),
		vulki.BindBuffer(1, out),
		vulki.BindBuffer(2, paramBuf),
		vulki.BindBuffer(3, countBuf),
	)
	if err != nil {
		t.Fatalf("bind fused windows: %v", err)
	}
	defer func() { _ = bindings.Close() }()

	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("new recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Upload(masks, 0, packed); err != nil {
		t.Fatalf("upload masks: %v", err)
	}
	if err := recorder.Update(paramBuf, 0, finderScanParams(width, height, channelMask, capacity, planeWords, geom, flags)); err != nil {
		t.Fatalf("upload params: %v", err)
	}
	if err := recorder.Update(countBuf, 0, make([]byte, finderWindowCounterCount*4)); err != nil {
		t.Fatalf("clear counters: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: uint32(geom.lineCount), Y: 3, Z: 1}); err != nil {
		t.Fatalf("dispatch fused windows: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run fused windows: %v", err)
	}

	rawCounts := make([]byte, finderWindowCounterCount*4)
	if err := countBuf.Download(rawCounts); err != nil {
		t.Fatalf("download counters: %v", err)
	}
	for i := range counts {
		counts[i] = binary.LittleEndian.Uint32(rawCounts[i*4:])
	}
	stored := min(int(counts[0]), capacity)
	raw := make([]byte, stored*finderWindowRecordWords*4)
	if stored > 0 {
		if err := out.DownloadAt(0, raw); err != nil {
			t.Fatalf("download survivors: %v", err)
		}
	}
	found = make([]finderWindow, stored)
	meta = make(map[finderWindow]finderRecordMeta, stored)
	for i := range found {
		at := i * finderWindowRecordWords * 4
		word := binary.LittleEndian.Uint32(raw[at:])
		found[i].key = word & (1<<finderEvidenceBits - 1)
		for k := range found[i].boundary {
			found[i].boundary[k] = binary.LittleEndian.Uint32(raw[at+(k+1)*4:])
		}
		meta[found[i]] = finderRecordMeta{
			evidence: word >> finderEvidenceBits,
			module:   math.Float32frombits(binary.LittleEndian.Uint32(raw[at+7*4:])),
		}
	}
	return found, meta, counts
}

// windowsFromBoundaries is the oracle for the fused prototype: every six
// consecutive boundaries of every line, classified by the same acceptance test
// and then by the same perpendicular cross-check. It reads the boundary lists
// prototypes 1 and 2 produce, so agreement between the two designs is checked
// against shared ground truth rather than against one of them being declared
// correct.
//
// The classification is three-way, not two. Acceptance is a ratio comparison
// evaluated in f32, and a run pattern such as 5-5-10-5-5 sits exactly on the
// tolerance: the device compiler is free to contract the division into a
// reciprocal multiply, which moves the last bit and flips the answer. Demanding
// that the host reproduce those bits would be pinning down float behaviour this
// project has deliberately left free. So a window near the tie is reported as
// undecided and the test permits either answer, while everything clear of the
// tie must match exactly - which is what actually exercises the seam carry.
//
// The cross-check widens what "near a tie" has to mean, because it addresses
// pixels rather than only comparing counts. A walk sample whose coordinate lands
// within a whisker of a pixel boundary can floor either way between the device's
// f32 and this walk, which changes a run length by one and can flip a verdict
// nowhere near a ratio tie. Such a window is undecided too.
func windowsFromBoundaries(boundaries [][]uint32, cross finderCrossOracle) (
	accepted, undecided []finderWindow,
	meta map[finderWindow]finderRecordMeta,
	square [2]int,
) {
	meta = make(map[finderWindow]finderRecordMeta)
	for key, list := range boundaries {
		for t := 0; t+5 < len(list); t++ {
			w := finderWindow{key: uint32(key)}
			copy(w.boundary[:], list[t:t+6])
			s1, s2, s3 := list[t+2]-list[t+1], list[t+3]-list[t+2], list[t+4]-list[t+3]
			along := float64(s1+s2+s3) / 3
			verdict := classifyFinderWindow(
				list[t+1]-list[t], s1, s2, s3, list[t+5]-list[t+4],
			)
			if verdict != windowRejected {
				crossed, expect := cross.record(w, along)
				verdict = weakest(verdict, crossed)
				if verdict == windowAccepted {
					meta[w] = expect
				}
			}
			switch verdict {
			case windowAccepted:
				accepted = append(accepted, w)
			case windowUndecided:
				undecided = append(undecided, w)
			default:
				continue
			}
			switch weakest(verdict, cross.diagonalRescued(w, along)) {
			case windowAccepted:
				square[0]++
			case windowUndecided:
				square[1]++
			}
		}
	}
	return accepted, undecided, meta, square
}

// weakest combines two verdicts the way the kernel's conjunction does: a clear
// rejection anywhere rejects, and an undecided anywhere leaves the whole thing
// undecided.
func weakest(a, b windowVerdict) windowVerdict {
	if a == windowRejected || b == windowRejected {
		return windowRejected
	}
	if a == windowUndecided || b == windowUndecided {
		return windowUndecided
	}
	return windowAccepted
}

// finderCrossOracle is the host model of shaders/finder_cross_check.wgsl. It
// walks in float32 rather than float64 for the same reason the shader does, so
// the two disagree only where the hardware is free to disagree with itself.
type finderCrossOracle struct {
	width, height int
	channel       int
	mask          func(x, y, channel int) bool
	geom          finderRunsGeometry
}

// pixelTieBand is how close to a pixel boundary a walk sample may land before
// its verdict is treated as unresolvable. It is far wider than an f32 ulp at
// these magnitudes, which is deliberate: the shader is free to contract its
// address arithmetic, and the point is to stop testing float behaviour that was
// never specified, not to model it.
const pixelTieBand = 1e-3

// resolves reports whether a coordinate floors to the same pixel however the
// arithmetic that produced it was rounded.
//
// **Landing exactly on a boundary is the unambiguous case, not the worst one.**
// At 0 and 45 degrees every step is a small integer or a half, which both f32
// and f64 hold exactly, so the two agree on every sample; treating that as a tie
// left those angles with nothing decided at all. What cannot be resolved is a
// coordinate that missed a boundary by a hair, since that is what accumulated
// rounding looks like.
func resolves(v float32) bool {
	frac := float64(v) - math.Floor(float64(v))
	return frac == 0 || (frac > pixelTieBand && frac < 1-pixelTieBand)
}

// at returns the mask bit under a point, 2 outside the frame, and whether the
// point sits clear of both pixel boundaries.
func (o finderCrossOracle) at(px, py float32) (int, bool) {
	clear := resolves(px) && resolves(py)
	x, y := math.Floor(float64(px)), math.Floor(float64(py))
	if x < 0 || x >= float64(o.width) || y < 0 || y >= float64(o.height) {
		return 2, clear
	}
	if o.mask(int(x), int(y), o.channel) {
		return 1, clear
	}
	return 0, clear
}

// walkSide is walk_side: three runs outwards, stopping at the third colour
// change, the frame edge or the cap.
func (o finderCrossOracle) walkSide(cx, cy, sx, sy float32, mid, cap int) (counts [3]int, stage int, clear bool) {
	prev := mid
	clear = true
	for i := 1; i <= cap; i++ {
		v, ok := o.at(cx+float32(i)*sx, cy+float32(i)*sy)
		clear = clear && ok
		if v > 1 {
			break
		}
		if v != prev {
			stage++
			if stage > 2 {
				break
			}
			prev = v
		}
		counts[stage]++
	}
	return counts, stage, clear
}

// layer is cross_layer: the module size the two halves imply, or a rejection,
// with the offset of the middle run's true centre in steps.
func (o finderCrossOracle) layer(cx, cy, ux, uy float32, along float64) (windowVerdict, float64, float32) {
	mid, midClear := o.at(cx, cy)
	if mid > 1 {
		return windowRejected, 0, 0
	}
	scale := float32(math.Hypot(float64(o.geom.dx), float64(o.geom.dy)))
	sx, sy := scale*ux, scale*uy
	// The oracle walks without the shader's per-run early exits, so the two agree
	// only if those exits really are outcome-preserving.
	cap := int(float32(along)*6) + 6
	back, backStage, backClear := o.walkSide(cx, cy, -sx, -sy, mid, cap)
	fwd, fwdStage, fwdClear := o.walkSide(cx, cy, sx, sy, mid, cap)
	clear := midClear && backClear && fwdClear
	if backStage < 2 || fwdStage < 2 {
		if clear {
			return windowRejected, 0, 0
		}
		return windowUndecided, 0, 0
	}
	s2 := back[0] + fwd[0] + 1
	verdict := classifyFinderWindow(
		uint32(back[2]), uint32(back[1]), uint32(s2), uint32(fwd[1]), uint32(fwd[2]))
	if !clear {
		verdict = weakest(verdict, windowUndecided)
	}
	return verdict, float64(back[1]+s2+fwd[1]) / 3, float32(fwd[0]-back[0]) / 2
}

// agrees is the shader's agrees: a walk has to find the signature and imply a
// module size within twice the along-line one.
func (o finderCrossOracle) agrees(verdict windowVerdict, measured, along float64) windowVerdict {
	if verdict == windowRejected {
		return windowRejected
	}
	switch limit := 2 * along; {
	case math.Abs(measured-limit) <= limit*4*1.1920929e-7:
		return windowUndecided
	case measured > limit:
		return windowRejected
	}
	return verdict
}

// centre is where the window's midpoint lands in the frame.
func (o finderCrossOracle) centre(w finderWindow) (float32, float32) {
	origin := lineOrigin(o.geom, w.key/3)
	mid := (float64(w.boundary[2]) + float64(w.boundary[3])) * 0.5
	return float32(origin[0] + mid*float64(o.geom.dx)), float32(origin[1] + mid*float64(o.geom.dy))
}

// walk runs one off-line direction through a window's centre and reports whether
// it agrees.
func (o finderCrossOracle) walk(w finderWindow, ux, uy float32, along float64) windowVerdict {
	cx, cy := o.centre(w)
	verdict, measured, _ := o.layer(cx, cy, ux, uy, along)
	return o.agrees(verdict, measured, along)
}

// confirm is cross_confirm: the walk repeated from the centre it implies, with
// the mean of the two module sizes the kernel stores.
func (o finderCrossOracle) confirm(w finderWindow, ux, uy float32, along float64) (windowVerdict, float64) {
	cx, cy := o.centre(w)
	scale := float32(math.Hypot(float64(o.geom.dx), float64(o.geom.dy)))
	verdict, first, offset := o.layer(cx, cy, ux, uy, along)
	if v := o.agrees(verdict, first, along); v != windowAccepted {
		return v, 0
	}
	verdict, again, _ := o.layer(cx+offset*scale*ux, cy+offset*scale*uy, ux, uy, along)
	if v := o.agrees(verdict, again, along); v != windowAccepted {
		return v, 0
	}
	return windowAccepted, (first + again) / 2
}

// record is what flush_block writes for a window it keeps: the verdict, which
// walk confirmed it, and that walk's module size. The order matters and is the
// kernel's - perpendicular, then the right diagonal, then the left - because the
// first walk to agree is the one recorded.
func (o finderCrossOracle) record(w finderWindow, along float64) (windowVerdict, finderRecordMeta) {
	dirs := o.directions()
	cx, cy := o.centre(w)
	verdict, measured, _ := o.layer(cx, cy, dirs[0][0], dirs[0][1], along)
	switch o.agrees(verdict, measured, along) {
	case windowAccepted:
		return windowAccepted, finderRecordMeta{evidence: 1, module: float32(measured)}
	case windowUndecided:
		return windowUndecided, finderRecordMeta{}
	}
	for i, d := range dirs[1:] {
		switch v, module := o.confirm(w, d[0], d[1], along); v {
		case windowAccepted:
			return windowAccepted, finderRecordMeta{evidence: uint32(2 + i), module: float32(module)}
		case windowUndecided:
			return windowUndecided, finderRecordMeta{}
		}
	}
	return windowRejected, finderRecordMeta{}
}

// directions is the perpendicular followed by the two diagonals, in the order
// flush_block tries them.
func (o finderCrossOracle) directions() [3][2]float32 {
	scale := float32(math.Hypot(float64(o.geom.dx), float64(o.geom.dy)))
	ux, uy := o.geom.dx/scale, o.geom.dy/scale
	unit := func(x, y float32) [2]float32 {
		n := float32(math.Hypot(float64(x), float64(y)))
		return [2]float32{x / n, y / n}
	}
	return [3][2]float32{
		{o.geom.nx, o.geom.ny},
		unit(ux+o.geom.nx, uy+o.geom.ny),
		unit(ux-o.geom.nx, uy-o.geom.ny),
	}
}

// rescue is the diagonal branch: either diagonal, confirmed twice.
func (o finderCrossOracle) rescue(w finderWindow, along float64) windowVerdict {
	best := windowRejected
	dirs := o.directions()
	for _, d := range dirs[1:] {
		switch v, _ := o.confirm(w, d[0], d[1], along); v {
		case windowAccepted:
			return windowAccepted
		case windowUndecided:
			best = windowUndecided
		}
	}
	return best
}

// diagonalRescued is the counted class: the perpendicular failed and a diagonal
// carried the candidate anyway.
//
// It is modelled here because this count drifted once without being noticed.
// Adding walk_side's early exits changed it and nothing failed, since the
// diagonals were then judged without the module-size bound those exits assume.
func (o finderCrossOracle) diagonalRescued(w finderWindow, along float64) windowVerdict {
	dirs := o.directions()
	switch o.walk(w, dirs[0][0], dirs[0][1], along) {
	case windowAccepted:
		return windowRejected
	case windowUndecided:
		return windowUndecided
	}
	return o.rescue(w, along)
}

// jabFinderMask reports the mask bit of a JAB finder pattern of the given module
// size, centred on the pattern's core, in module coordinates relative to it.
//
// **A JAB finder is not a concentric ring target.** Per ISO/IEC 23634:2022 4.3.7
// it is two equal 3x3 square references joined at a single overlapping module,
// the core, laid out along one diagonal - see the note on the finder types in
// finderpattern.go, which exists because assuming rings has repeatedly produced
// wrong reasoning here. A ring fixture was used in this test and in the
// benchmark frame, and it is isotropic in a way the real shape is not: the
// joining diagonal carries the signature and the other one runs straight out of
// the pattern into background.
//
// The layer map is the encoder's: placePrimaryFinderPatterns writes layer k at
// the outer edge of a (k+1) square in each reference, mirrored through the core,
// so a module's layer is its Chebyshev distance from the core *within its own
// reference*. Layers 0 and 2 are dark and layer 1 is light, which is what a
// binarized channel separates; the quiet zone around it is light too.
//
// **Guessing this shape has now failed three times, so it is derived rather than
// described.** A first fixture cleared only the two reference centres, leaving
// the four layer-1 modules that sit beside the core dark, which is not a finder
// and not what the encoder writes. Before that the outer layer and the quiet
// zone were both light, leaving three runs across the whole pattern and no
// n-1-1-1-m window anywhere in the fixture.
//
// What the real map gives, and what every isotropic stand-in got wrong: the
// signature runs along the core's row, the core's column and the joining
// diagonal, all 1-1-1-1-1. Along the other diagonal there is only the core, and
// along a row that misses the core there is no signature at all.
func jabFinderMask(dx, dy, module int, mirrored bool) bool {
	if mirrored {
		dy = -dy
	}
	mx, my := floorDiv(dx, module), floorDiv(dy, module)
	var layer int
	switch {
	case mx <= 0 && my <= 0 && mx >= -2 && my >= -2:
		layer = max(-mx, -my)
	case mx >= 0 && my >= 0 && mx <= 2 && my <= 2:
		layer = max(mx, my)
	default:
		return false
	}
	return layer != 1
}

// floorDiv divides toward negative infinity, so module coordinates are
// continuous across the pattern's centre rather than folding at zero.
func floorDiv(a, b int) int {
	q := a / b
	if a%b != 0 && (a < 0) != (b < 0) {
		q--
	}
	return q
}

// lineOrigin is line_origin: where the line at this index starts.
func lineOrigin(geom finderRunsGeometry, line uint32) [2]float64 {
	q := float64(geom.qLo) + float64(line)*float64(geom.qStep)
	return [2]float64{q * float64(geom.nx), q * float64(geom.ny)}
}

type windowVerdict int

const (
	windowRejected windowVerdict = iota
	windowAccepted
	windowUndecided
)

// classifyFinderWindow evaluates checkPatternCross in float64 and reports
// undecided when any comparison is closer to its threshold than f32 can
// resolve.
func classifyFinderWindow(s0, s1, s2, s3, s4 uint32) windowVerdict {
	if s1 == 0 || s2 == 0 || s3 == 0 {
		return windowRejected
	}
	layer := float64(s1+s2+s3) / 3
	tol := layer / 2
	// A few f32 ulps at the magnitude the comparison is made, which covers a
	// reciprocal multiply substituted for the division on either side.
	eps := tol * 4 * 1.1920929e-7
	rejected, undecided := false, false
	// less records that value < limit is required to accept.
	less := func(value, limit float64) {
		switch {
		case math.Abs(value-limit) <= eps:
			undecided = true
		case value >= limit:
			rejected = true
		}
	}
	less(math.Abs(layer-float64(s1)), tol)
	less(math.Abs(layer-float64(s2)), tol)
	less(math.Abs(layer-float64(s3)), tol)
	less(0.5*tol, float64(s0))
	less(0.5*tol, float64(s4))
	less(math.Abs(float64(s1)-float64(s3)), tol)
	switch {
	case rejected:
		return windowRejected
	case undecided:
		return windowUndecided
	default:
		return windowAccepted
	}
}

// The fused prototype trades the boundary buffer for workgroup state carried
// across block seams, which is exactly where a windowing kernel breaks: a
// window spanning a seam needs the five boundaries that preceded the block's
// own first one. Carrying four instead of five drops one window per seam, and
// nothing about the kernel's output shape would reveal it. So its survivors are
// compared against the windows enumerated from the boundary lists, over angled
// sweeps whose lines cross many seams.
func TestGPUFinderWindowsMatchBoundaryWindows(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)

	// A grid of real JAB finder patterns, so the cross-check is exercised against
	// the shape it will actually meet. Two earlier fixtures were wrong in the
	// same direction, each more isotropic than a finder is: a checkerboard, whose
	// anti-diagonal has no signature at all and left the 45 degree cases deciding
	// nothing, and then a concentric ring target, which carries the signature
	// along every line through it and so could never show a difference between
	// the perpendicular and the diagonals.
	const module, pitch = 5, 61
	mask := func(x, y int) bool {
		// Alternating cells draw the two joining-diagonal classes, so both
		// diagonal labels appear in the records.
		mirrored := (x/pitch+y/pitch)%2 == 1
		return jabFinderMask(x%pitch-pitch/2, y%pitch-pitch/2, module, mirrored)
	}
	// Ballot support and a full-subgroup guarantee are separate capabilities and
	// both are needed, so the gate is the same one production selection uses
	// rather than a partitioning probe that a ballot-less device would pass.
	subgroups, err := kernels.subgroupKernelsUsable()
	if err != nil {
		t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}

	for _, variant := range finderWindowVariants(t, kernels) {
		for _, deg := range []float64{0, 15, 45, 75} {
			t.Run(fmt.Sprintf("%s/%.0f degrees", variant.name, deg), func(t *testing.T) {
				if variant.subgroup && !subgroups {
					t.Skip("this adapter cannot build the ballot kernels; the portable twin is its route")
				}
				const width, height = 400, 400
				layout := variant.layout
				geom := sweepGeometry(width, height, deg, 11)
				packed, planeWords := packFinderRunsMasks(layout, width, height, func(x, y, channel int) bool {
					return channel == 1 && mask(x, y)
				})
				capacity := geom.lineLength + 8
				boundaries, _ := runFinderRuns(t, device, kernels,
					finderRunsVariant{"hillis", layout, (*gpuDecodeKernels).finderRunsHillis},
					width, height, 1<<1, capacity, packed, planeWords, geom)
				cross := finderCrossOracle{
					width: width, height: height, channel: 1, geom: geom,
					mask: func(x, y, channel int) bool { return channel == 1 && mask(x, y) },
				}
				want, undecided, wantMeta, square := windowsFromBoundaries(boundaries, cross)
				if len(want) == 0 {
					t.Fatalf("the case accepted no windows, so it is not testing agreement (%d undecided)", len(undecided))
				}

				got, gotMeta, counts := runFinderWindows(t, device, kernels, variant,
					width, height, 1<<1, len(want)+len(undecided)+64, packed, planeWords, geom, 0)
				assertWindowsCover(t, got, want, undecided)
				assertRecordMeta(t, gotMeta, wantMeta)
				if int(counts[0]) != len(got) {
					t.Fatalf("fused kernel reported %d candidates but wrote %d", counts[0], len(got))
				}
				// Each counter is a subset of the one before it, which is the only
				// relationship between them the host can check without repeating
				// the walks.
				if counts[1] > counts[0] || counts[2] > counts[0] {
					t.Fatalf("nested counts are not nested: %v", counts)
				}
				t.Logf("%d windows accepted along the line, %d survive the cross-check, %d with inner runs >= 3, %d rescued by a diagonal",
					counts[3], counts[0], counts[1], counts[2])
				if counts[0] > counts[3] {
					t.Fatalf("more candidates survived the cross-check than were accepted before it: %v", counts)
				}
				if int(counts[2]) < square[0] || int(counts[2]) > square[0]+square[1] {
					t.Fatalf("diagonal-rescued count %d is outside the oracle's %d to %d",
						counts[2], square[0], square[0]+square[1])
				}
			})
		}
	}
}

// assertWindowsCover checks the fused kernel emitted every clearly accepted
// window and nothing outside the accepted-or-undecided set. Comparison is by
// set because the kernel appends in whatever order its blocks reserve slots.
func assertWindowsCover(t *testing.T, got, want, undecided []finderWindow) {
	t.Helper()
	index := func(list []finderWindow) map[finderWindow]bool {
		out := make(map[finderWindow]bool, len(list))
		for _, w := range list {
			out[w] = true
		}
		return out
	}
	gotIndex, wantIndex, tieIndex := index(got), index(want), index(undecided)
	if len(got) != len(gotIndex) {
		t.Fatalf("fused kernel emitted %d records for %d distinct windows, so it duplicated some", len(got), len(gotIndex))
	}
	for w := range wantIndex {
		if !gotIndex[w] {
			t.Fatalf("window %v is clearly accepted but the fused kernel did not emit it", w)
		}
	}
	for w := range gotIndex {
		if !wantIndex[w] && !tieIndex[w] {
			t.Fatalf("fused kernel emitted window %v, which the boundary oracle rejects outright", w)
		}
	}
}

// The cross-walk bound has to come from the frame, not from the scan line's
// sample budget. line_length is how far a line runs in the *scan* direction, and
// horizontalGeometry sets it to the width, so on a tall frame a perpendicular
// walk can need more samples than line_length while still being nowhere near
// leaving the image. Every other fused case sweeps at an angle where
// line_length is already width plus height, so none of them can tell the two
// bounds apart.
//
// **The reach that matters comes from where the centre lands, not from the
// window's own size.** A walk starts at the along-line window's midpoint, which
// says nothing about where that point sits inside the off-line run through it.
// Put it at one end of a long middle run and one side must cross that whole run
// before it even reaches the first transition.
//
// Here the window is 26-100-100-100-26, needing 352 samples of line. The
// perpendicular through its centre has physical runs 60-170-242-168-60, entered
// at the far end of the 242: the backward side spends 239 samples finishing that
// run and 170 more crossing the next, so it only enters the outer run at step
// 410. The old bound stopped at 352, inside the adjacent run, and the window was
// lost. The outer counts come back as 52 rather than 60 because the walk caps
// them once the verdict is settled, which is why the measured signature is
// 52-170-242-168-52 and its module size 193.33.
//
// **Every number above is enumerated from the mask below, not summarised.** A
// first attempt at this test was abandoned on the mistaken conclusion that no
// such case existed; the description that replaced it was wrong about the runs;
// and the assertion that replaced *that* checked only the inner-run sum, which
// several different geometries share. Prose about a fixture drifts, so the runs
// are counted and the two properties the fixture exists for are stated as
// arithmetic.
func TestGPUFinderWindowsWalkPastTheLineBudget(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})

	const (
		outer, inner = 26, 100
		width        = 2*outer + 3*inner
		centreX      = outer + inner + inner/2
		vOuter       = 60
		vMid         = 240
		vInner       = 170
		rowY         = vOuter + vInner + vMid
		height       = rowY + vInner + vOuter + 16
		band         = 2
	)
	horizontal := func(x int) bool {
		switch {
		case x < outer, x >= outer+inner && x < outer+2*inner, x >= outer+3*inner:
			return true
		}
		return false
	}
	// The middle run ends at rowY, so a walk starting there crosses all of it.
	vertical := func(y int) bool {
		switch d := y - rowY; {
		case d >= -(vMid-1) && d <= 0, d >= vInner+1 && d <= vInner+vOuter:
			return true
		case d >= -(vMid-1+vInner+vOuter) && d < -(vMid-1+vInner):
			return true
		}
		return false
	}
	mask := func(x, y, channel int) bool {
		switch {
		case channel != 1:
			return false
		case y >= rowY-band && y <= rowY+band:
			return horizontal(x)
		case x >= centreX-band && x <= centreX+band:
			return vertical(y)
		}
		return false
	}

	// The two properties the fixture exists for, as arithmetic rather than
	// commentary. If either stops holding the test is no longer the regression
	// it claims to be, whatever else still passes.
	if outerCap := inner/2 + 2; outerCap >= vOuter {
		t.Fatalf("the outer runs are %d and the walk caps them at %d, so nothing is capped", vOuter, outerCap)
	}
	// The walk spends vMid-1 steps finishing the middle run and vInner crossing
	// the next, so the outer run's first sample is step vMid+vInner. The bound
	// has to fall short of *that* step, not of the one before it: a line budget
	// equal to the last adjacent-run sample still never reaches the outer run,
	// and is a perfectly good fixture.
	if reach := vMid + vInner; reach <= width {
		t.Fatalf("the far side reaches the outer run at step %d, within the %d-sample line budget", reach, width)
	}
	// **The expected runs are literals on purpose.** Writing them in terms of the
	// constants above makes the assertion a tautology: change vMid and vInner
	// together and both sides move, which is exactly the drift this is here to
	// catch. The middle run is 242 rather than 240, and the run after it 168
	// rather than 170, because the five rows of the horizontal band are dark at
	// this column and extend the middle run past rowY.
	assertMaskRuns(t, "row "+strconv.Itoa(rowY), width,
		func(x int) bool { return mask(x, rowY, 1) },
		[]int{26, 100, 100, 100, 26})
	assertMaskRuns(t, "column "+strconv.Itoa(centreX), height,
		func(y int) bool { return mask(centreX, y, 1) },
		[]int{1, 60, 170, 242, 168, 60, 15})

	geom := horizontalGeometry(width, height)
	for _, variant := range finderWindowVariants(t, kernels) {
		t.Run(variant.name, func(t *testing.T) {
			if variant.subgroup {
				// An adapter that cannot run the ballot kernels skips; one that
				// says it can and then fails is a defect and must not.
				usable, err := kernels.subgroupKernelsUsable()
				if err != nil {
					t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
				}
				if !usable {
					t.Skip("this adapter cannot build the ballot kernels; the portable twin is its route")
				}
			}
			packed, planeWords := packFinderRunsMasks(variant.layout, width, height, mask)
			cross := finderCrossOracle{
				width: width, height: height, channel: 1, geom: geom, mask: mask,
			}
			boundaries, _ := runFinderRuns(t, device, kernels,
				finderRunsVariant{"hillis", variant.layout, (*gpuDecodeKernels).finderRunsHillis},
				width, height, 1<<1, width+8, packed, planeWords, geom)
			want, undecided, _, _ := windowsFromBoundaries(boundaries, cross)
			if len(want) == 0 {
				t.Fatalf("the fixture accepts no window, so it cannot show the bound (%d undecided)", len(undecided))
			}
			got, meta, _ := runFinderWindows(t, device, kernels, variant,
				width, height, 1<<1, len(want)+len(undecided)+8, packed, planeWords, geom, 0)
			assertWindowsCover(t, got, want, undecided)

			// The window on the row the perpendicular was built around, with the
			// module size that perpendicular implies: (170 + 242 + 168) / 3.
			centre := finderWindow{
				// The fixture is drawn on channel 1, and a key is line * 3 + channel.
				key:      uint32(rowY*3 + 1),
				boundary: [6]uint32{0, outer, outer + inner, outer + 2*inner, outer + 3*inner, width},
			}
			actual, ok := meta[centre]
			if !ok {
				t.Fatalf("the window the fixture is built around is missing: %v", centre)
			}
			if actual.evidence != 1 {
				t.Fatalf("the long walk was the perpendicular, but the record names walk %d", actual.evidence)
			}
			if want := float32(580) / 3; !(math.Abs(float64(actual.module-want)) <= 1e-3) {
				t.Fatalf("perpendicular module size is %g, so the fixture is not the shape described: want %g",
					actual.module, want)
			}
		})
	}
}

// assertMaskRuns enumerates a fixture's run lengths along one line and holds
// them to what its documentation claims. A sum or a module size is not enough:
// several different run sequences share one, so a fixture can drift into a
// different shape while every derived quantity still matches.
func assertMaskRuns(t *testing.T, where string, n int, at func(int) bool, want []int) {
	t.Helper()
	var got []int
	prev, count := at(0), 1
	for i := 1; i < n; i++ {
		if v := at(i); v != prev {
			got = append(got, count)
			prev, count = v, 1
			continue
		}
		count++
	}
	got = append(got, count)
	if !slices.Equal(got, want) {
		t.Fatalf("%s has runs %v, but the fixture is documented as %v", where, got, want)
	}
}

// assertRecordMeta holds every clearly accepted window's metadata to what the
// oracle's own walks measured. Without it the evidence field and the stored
// module size are unchecked: a kernel that labelled both diagonals the same,
// swapped them, or wrote a meaningless module size would pass everything else
// here, because none of it reaches the window set the coverage check compares.
//
// Only windows the oracle decided clearly are held, for the same reason the set
// comparison tolerates ties.
func assertRecordMeta(t *testing.T, got, want map[finderWindow]finderRecordMeta) {
	t.Helper()
	var seen [4]int
	for _, actual := range got {
		seen[actual.evidence]++
	}
	// A record with no confirming walk cannot exist, and a fixture that never
	// produces one of the two diagonal labels would leave the label it does
	// produce indistinguishable from a kernel that only ever writes one.
	if seen[0] > 0 {
		t.Fatalf("%d records were emitted with no walk confirming them", seen[0])
	}
	if seen[2] == 0 || seen[3] == 0 {
		t.Fatalf("the fixture produced %v evidence labels, so the two diagonals are not distinguished", seen)
	}
	for w, expect := range want {
		actual, ok := got[w]
		if !ok {
			t.Fatalf("window %v is clearly accepted but the kernel emitted no record", w)
		}
		if actual.evidence != expect.evidence {
			t.Fatalf("window %v was confirmed by walk %d on the device and walk %d in the oracle",
				w, actual.evidence, expect.evidence)
		}
		// The module size is a mean of run counts, so it lands on an exact
		// multiple of a third and the two sides agree outright unless a walk
		// differed.
		//
		// The comparison is written as a required closeness rather than a
		// forbidden distance, because every comparison against a NaN is false: a
		// shader storing one would have satisfied `difference > tolerance` being
		// false and passed the very check meant to catch meaningless metadata.
		if !(math.Abs(float64(actual.module-expect.module)) <= 1e-3) {
			t.Fatalf("window %v records module size %g, oracle measured %g",
				w, actual.module, expect.module)
		}
	}
}

// sweepGeometry builds the same line family sweepDirection walks: lines at the
// given angle, spaced step apart perpendicular to it, indexed so that the
// frame's corners bound the perpendicular offset.
func sweepGeometry(width, height int, deg float64, step float64) finderRunsGeometry {
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
	return finderRunsGeometry{
		dx: float32(dir.dx), dy: float32(dir.dy),
		nx: float32(nx), ny: float32(ny),
		qLo: float32(qLo), qStep: float32(step),
		lineCount:  int((qHi-qLo)/step) + 1,
		lineLength: width + height,
	}
}

// finderScanParams serializes the parameter block every directional prototype
// reads.
func finderScanParams(
	width, height int,
	channelMask uint32,
	capacity, planeWords int,
	geom finderRunsGeometry,
	flags uint32,
) []byte {
	params := make([]byte, finderScanParamsBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	binary.LittleEndian.PutUint32(params[8:], channelMask)
	binary.LittleEndian.PutUint32(params[12:], uint32(geom.lineLength))
	binary.LittleEndian.PutUint32(params[16:], math.Float32bits(geom.dx))
	binary.LittleEndian.PutUint32(params[20:], math.Float32bits(geom.dy))
	binary.LittleEndian.PutUint32(params[24:], math.Float32bits(geom.nx))
	binary.LittleEndian.PutUint32(params[28:], math.Float32bits(geom.ny))
	binary.LittleEndian.PutUint32(params[32:], math.Float32bits(geom.qLo))
	binary.LittleEndian.PutUint32(params[36:], math.Float32bits(geom.qStep))
	binary.LittleEndian.PutUint32(params[40:], uint32(geom.lineCount))
	binary.LittleEndian.PutUint32(params[44:], uint32(capacity))
	binary.LittleEndian.PutUint32(params[48:], uint32(planeWords))
	binary.LittleEndian.PutUint32(params[52:], flags)
	return params
}

// FLAG_EMIT_UNCONFIRMED has to make the pre-cross-check candidate *set*
// readable, not just its size. counters[3] already gives the size, and a size
// cannot be diffed against another generator's candidates: two runs with equal
// totals can have exchanged them. Measuring the cross-check's recall against the
// CPU sweep needs the set, which is what this mode emits.
//
// Everything here is a relation between two runs of the same kernel over the
// same masks, so it needs no second oracle: the flag must change which windows
// are recorded and nothing else about the verdicts.
func TestGPUFinderWindowsEmitUnconfirmedExposesTheRejectedSet(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})

	subgroups, err := kernels.subgroupKernelsUsable()
	if err != nil {
		t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}

	// A real finder pattern, so the rejected population is the one a capture
	// produces around a symbol rather than an artifact of a drawn shape.
	const width, height, module = 400, 400, 9
	mask := func(x, y int) bool {
		return jabFinderMask(x-width/2, y-height/2, module, false)
	}

	for _, variant := range finderWindowVariants(t, kernels) {
		t.Run(variant.name, func(t *testing.T) {
			if variant.subgroup && !subgroups {
				t.Skip("this adapter cannot build the ballot kernels; the portable twin is its route")
			}
			layout := variant.layout
			geom := sweepGeometry(width, height, 45, 3)
			packed, planeWords := packFinderRunsMasks(layout, width, height, func(x, y, channel int) bool {
				return channel == 1 && mask(x, y)
			})
			capacity := 4096

			filtered, filteredMeta, filteredCounts := runFinderWindows(t, device, kernels, variant,
				width, height, 1<<1, capacity, packed, planeWords, geom, 0)
			all, meta, allCounts := runFinderWindows(t, device, kernels, variant,
				width, height, 1<<1, capacity, packed, planeWords, geom, finderScanEmitUnconfirmed)

			if allCounts[3] == filteredCounts[0] {
				t.Fatalf("the fixture has nothing the cross-check rejects, so it cannot show the mode (%v)", allCounts)
			}
			if int(allCounts[0]) > capacity || int(filteredCounts[0]) > capacity {
				t.Fatalf("the record buffer overflowed, so the sets below are truncated: %v %v", allCounts, filteredCounts)
			}
			if allCounts[0] != allCounts[3] {
				t.Fatalf("unfiltered run recorded %d of %d accepted windows", allCounts[0], allCounts[3])
			}
			// The flag selects what is written. It must not move a verdict, so
			// the along-line population and both cross-check subsets are equal
			// across the two runs.
			for _, i := range []int{1, 2, 3} {
				if allCounts[i] != filteredCounts[i] {
					t.Fatalf("counter %d moved with the flag: %v against %v", i, allCounts, filteredCounts)
				}
			}

			// Records are compared as sets, and the sets have to be sets: a
			// duplicated record would let a membership check pass while the
			// populations differed.
			if len(filtered) != len(filteredMeta) || len(all) != len(meta) {
				t.Fatalf("a run emitted duplicate records: %d of %d filtered, %d of %d unfiltered",
					len(filteredMeta), len(filtered), len(meta), len(all))
			}

			// Exact equality, not containment plus a count. Containment with a
			// matching confirmed *total* is satisfied by a symmetric swap - the
			// filtered run keeping A and B while the unfiltered run labels A and
			// C - which would be the kernel deciding two different things under
			// one flag, precisely what this test exists to exclude.
			confirmed := make(map[finderWindow]finderRecordMeta, len(filtered))
			for _, w := range all {
				m := meta[w]
				if m.evidence != 0 {
					confirmed[w] = m
					continue
				}
				// A rejected record carries no measurement. Leaving the last
				// failed walk's return in it would make it look measured.
				if !(m.module < 0) {
					t.Fatalf("unconfirmed window %v carries module size %v", w, m.module)
				}
			}
			if len(confirmed) != len(filtered) {
				t.Fatalf("unfiltered run labelled %d records confirmed, filtered run recorded %d", len(confirmed), len(filtered))
			}
			for _, w := range filtered {
				m, ok := confirmed[w]
				if !ok {
					t.Fatalf("window %v survived the cross-check but the unfiltered run did not label it confirmed", w)
				}
				// The verdict itself has to survive the flag, not only the
				// membership: a record that changed which walk confirmed it, or
				// what that walk measured, is a different answer.
				if m != filteredMeta[w] {
					t.Fatalf("window %v is %+v unfiltered and %+v filtered", w, m, filteredMeta[w])
				}
			}
		})
	}
}

// FLAG_SKIP_CROSS_CHECK is what the route dispatches with, so its contract needs
// pinning directly rather than inferred from the unfiltered mode: no walk runs,
// every accepted window is a record, and no record claims a verdict.
//
// Its equality with the unfiltered mode's record set is the load-bearing part.
// The route is a candidate generator, so a mode that quietly kept filtering, or
// one that emitted a different population, would change what the host chain is
// offered while every outcome table still passed.
func TestGPUFinderWindowsSkipCrossCheckEmitsEveryWindow(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})

	subgroups, err := kernels.subgroupKernelsUsable()
	if err != nil {
		t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}

	const width, height, module = 400, 400, 9
	mask := func(x, y int) bool {
		return jabFinderMask(x-width/2, y-height/2, module, false)
	}

	for _, variant := range finderWindowVariants(t, kernels) {
		t.Run(variant.name, func(t *testing.T) {
			if variant.subgroup && !subgroups {
				t.Skip("this adapter cannot build the ballot kernels; the portable twin is its route")
			}
			layout := variant.layout
			geom := sweepGeometry(width, height, 45, 3)
			packed, planeWords := packFinderRunsMasks(layout, width, height, func(x, y, channel int) bool {
				return channel == 1 && mask(x, y)
			})
			const capacity = 4096

			skipped, skippedMeta, skippedCounts := runFinderWindows(t, device, kernels, variant,
				width, height, 1<<1, capacity, packed, planeWords, geom, finderScanSkipCrossCheck)
			walked, _, walkedCounts := runFinderWindows(t, device, kernels, variant,
				width, height, 1<<1, capacity, packed, planeWords, geom, finderScanEmitUnconfirmed)

			if int(skippedCounts[0]) > capacity {
				t.Fatalf("record buffer overflowed at %d of %d", skippedCounts[0], capacity)
			}
			if skippedCounts[0] != skippedCounts[3] {
				t.Fatalf("skip mode recorded %d of %d accepted windows", skippedCounts[0], skippedCounts[3])
			}
			if skippedCounts[3] != walkedCounts[3] {
				t.Fatalf("the along-line population moved with the flag: %d against %d",
					skippedCounts[3], walkedCounts[3])
			}
			// No walk ran, so nothing can have been confirmed. Nonzero here
			// would mean the flag was ignored and the walks ran anyway.
			if skippedCounts[1] != 0 || skippedCounts[2] != 0 {
				t.Fatalf("skip mode reported cross-check verdicts: %v", skippedCounts)
			}
			if walkedCounts[1] == 0 {
				t.Fatal("the fixture confirms nothing even with the walks on, so the comparison is empty")
			}

			index := make(map[finderWindow]bool, len(walked))
			for _, w := range walked {
				index[w] = true
			}
			if len(skipped) != len(walked) || len(skippedMeta) != len(skipped) {
				t.Fatalf("skip mode wrote %d records for %d windows, walked mode wrote %d",
					len(skipped), len(skippedMeta), len(walked))
			}
			for _, w := range skipped {
				if !index[w] {
					t.Fatalf("skip mode emitted window %v that the walked run did not", w)
				}
				m := skippedMeta[w]
				if m.evidence != 0 {
					t.Fatalf("window %v claims evidence %d with no walk run", w, m.evidence)
				}
				if !(m.module < 0) {
					t.Fatalf("window %v claims module size %v with no walk run", w, m.module)
				}
			}
		})
	}
}
