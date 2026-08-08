//go:build !js

package detect

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

const (
	gpuRGBHistogramBytes = 3 * 256 * 4
	gpuRGBBoundsBytes    = 8 * 4
)

type gpuResidentInputBindings struct {
	histogram *vulki.BindingSet
	balance   *vulki.BindingSet
}

type gpuResidentPreparedBindings struct {
	blocks     *vulki.BindingSet
	classifier *vulki.BindingSet
}

// gpuResidentBinarizer consumes an image buffer that already belongs to its
// borrowed device. Histogram balancing, scale-adaptive block statistics and
// the fused classifier/filter/packer remain on that device; only packed masks
// cross back to the host. Each route context owns one instance, so concurrent
// routes never share its scratch buffers or binding sets.
type gpuResidentBinarizer struct {
	mu     sync.Mutex
	closed bool

	// generation counts binarization passes. Each pass's channel materializer
	// captures the generation at recording time and refuses to expand the
	// shared packed-mask host buffer after a later pass overwrote it.
	generation uint64

	// lazyChannels identifies the latest pass's shape-only channel bitmaps,
	// so a mask snapshot can prove it copies the packed words of the pass it
	// was asked about rather than a later pass's silently different masks.
	lazyChannels [3]*core.Bitmap

	// masksOnHost says the current pass's packed words have already been
	// fetched, so a second host consumer reuses them instead of transferring
	// the same masks again.
	masksOnHost bool

	// rowStride mirrors the consumer's finder walk spacing so the chain fold
	// skips exactly the rows the walk would.
	rowStride int

	device      *vulki.Device
	kernels     *gpuDecodeKernels
	ownsKernels bool
	binarizer   *gpuBinarizer

	histogram *vulki.Buffer
	bounds    *vulki.Buffer
	balanced  *vulki.Buffer

	sampleResult      *vulki.Buffer
	sampleParams      *vulki.Buffer
	moduleCountResult *vulki.Buffer
	moduleCountParams *vulki.Buffer
	alignCells        *vulki.Buffer
	alignParams       *vulki.Buffer
	alignTiles        *vulki.Buffer
	ldpcRows          *vulki.Buffer
	ldpcBits          *vulki.Buffer
	ldpcParams        *vulki.Buffer
	ldpcNet           *vulki.Buffer

	payloadParams      *vulki.Buffer
	payloadMap         *vulki.Buffer
	payloadPermutation *vulki.Buffer

	metadataParams *vulki.Buffer
	metadataRecord *vulki.Buffer

	foldParams     *vulki.Buffer
	foldCandidates *vulki.Buffer
	foldPatterns   *vulki.Buffer
	foldRecord     *vulki.Buffer

	// sampledGrid is the module grid the sampler most recently produced. The
	// payload chain reads that grid where it lies, so a correction asked about
	// any other sample - a cached alignment resample, another route's symbol -
	// must be declined rather than answered from the wrong modules.
	sampledGrid *core.Bitmap

	// permutationLength and permutationGenerator are what the resident
	// deinterleaving permutation was built for. The shuffle depends on nothing
	// else, so a correction that matches both reuses the table. A zero length
	// means it holds nothing usable.
	permutationLength    int
	permutationGenerator uint32

	histogramKernel       *vulki.Kernel
	boundsKernel          *vulki.Kernel
	balanceKernel         *vulki.Kernel
	blocksKernel          *vulki.Kernel
	sampleKernel          *vulki.Kernel
	moduleCountKernel     *vulki.Kernel
	alignKernel           *vulki.Kernel
	ldpcKernel            *vulki.Kernel
	payloadMapKernel      *vulki.Kernel
	payloadPermuteKernel  *vulki.Kernel
	payloadBitsKernel     *vulki.Kernel
	metadataPart1Kernel   *vulki.Kernel
	metadataPaletteKernel *vulki.Kernel
	metadataPart2Kernel   *vulki.Kernel
	metadataFinishKernel  *vulki.Kernel
	foldKernel            *vulki.Kernel
	sortKernel            *vulki.Kernel

	metadataPart1Bindings   *vulki.BindingSet
	metadataPaletteBindings *vulki.BindingSet
	metadataPart2Bindings   *vulki.BindingSet
	metadataFinishBindings  *vulki.BindingSet
	foldBindings            *vulki.BindingSet
	sortBindings            *vulki.BindingSet
	sampleBindings          *vulki.BindingSet
	moduleCountBindings     *vulki.BindingSet
	alignBindings           *vulki.BindingSet
	ldpcBindings            *vulki.BindingSet
	payloadMapBindings      *vulki.BindingSet
	payloadPermuteBindings  *vulki.BindingSet
	payloadBitsBindings     *vulki.BindingSet
	boundsBindings          *vulki.BindingSet
	inputBindings           map[*vulki.Buffer]gpuResidentInputBindings
	preparedBindings        map[*vulki.Buffer]gpuResidentPreparedBindings
}

func newGPUResidentBinarizerWithDevice(
	device *vulki.Device,
	maxWidth, maxHeight int,
) (*gpuResidentBinarizer, error) {
	kernels := newGPUDecodeKernels(device)
	resident, err := newGPUResidentBinarizerWithKernels(device, kernels, maxWidth, maxHeight)
	if err == nil {
		// A standalone resident binarizer compiles its chain kernels up
		// front rather than warming them in the background, so its first pass
		// already replays instead of spending one on the CPU twin.
		if err = kernels.compileFinderChains(); err != nil {
			_ = resident.Close()
		}
	}
	if err != nil {
		_ = kernels.Close()
		return nil, err
	}
	resident.ownsKernels = true
	return resident, nil
}

func newGPUResidentBinarizerWithKernels(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	maxWidth, maxHeight int,
) (*gpuResidentBinarizer, error) {
	binarizer, err := newGPUBinarizerPipelineWithDevice(device, kernels, maxWidth, maxHeight, false)
	if err != nil {
		return nil, err
	}
	resident := &gpuResidentBinarizer{
		device:           device,
		kernels:          kernels,
		binarizer:        binarizer,
		inputBindings:    make(map[*vulki.Buffer]gpuResidentInputBindings),
		preparedBindings: make(map[*vulki.Buffer]gpuResidentPreparedBindings),
	}
	if err := resident.initialize(); err != nil {
		_ = resident.closeResources()
		return nil, err
	}
	return resident, nil
}

func (resident *gpuResidentBinarizer) initialize() error {
	area := uint64(resident.binarizer.maxWidth) * uint64(resident.binarizer.maxHeight)
	var err error
	resident.histogram, err = resident.device.NewBuffer(gpuRGBHistogramBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU RGB histogram: %w", err)
	}
	resident.bounds, err = resident.device.NewBuffer(gpuRGBBoundsBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU RGB bounds: %w", err)
	}
	resident.balanced, err = resident.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU balanced image: %w", err)
	}

	resident.histogramKernel, err = resident.kernels.histogramRGB()
	if err != nil {
		return err
	}
	resident.boundsKernel, err = resident.kernels.histogramBounds()
	if err != nil {
		return err
	}
	resident.balanceKernel, err = resident.kernels.balanceRGB()
	if err != nil {
		return err
	}
	resident.blocksKernel, err = resident.kernels.blockThresholds()
	if err != nil {
		return err
	}

	resident.boundsBindings, err = resident.boundsKernel.NewBindings(
		vulki.BindBuffer(0, resident.histogram),
		vulki.BindBuffer(1, resident.bounds),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU histogram bounds: %w", err)
	}
	// The chain's colour stage samples the balanced image directly, which is
	// what lets a resident route decide the source-colour signal without the
	// host downloading the frame to answer it per candidate.
	resident.binarizer.colorSource = resident.balanced
	if _, err := resident.preparedBindingsFor(resident.balanced); err != nil {
		return err
	}
	if err := resident.initializeSampler(); err != nil {
		return err
	}
	if err := resident.initializeModuleCount(); err != nil {
		return err
	}
	if err := resident.initializeAlignment(); err != nil {
		return err
	}
	if err := resident.initializeLDPC(); err != nil {
		return err
	}
	if err := resident.initializePayload(); err != nil {
		return err
	}
	if err := resident.initializeMetadata(); err != nil {
		return err
	}
	return resident.initializeFinderFold()
}

func (resident *gpuResidentBinarizer) bindingsFor(
	input *vulki.Buffer,
) (gpuResidentInputBindings, error) {
	if bindings, ok := resident.inputBindings[input]; ok {
		return bindings, nil
	}
	var bindings gpuResidentInputBindings
	var err error
	bindings.histogram, err = resident.histogramKernel.NewBindings(
		vulki.BindBuffer(0, input),
		vulki.BindBuffer(1, resident.histogram),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		return bindings, fmt.Errorf("jabcode: bind resident GPU RGB histogram input: %w", err)
	}
	bindings.balance, err = resident.balanceKernel.NewBindings(
		vulki.BindBuffer(0, input),
		vulki.BindBuffer(1, resident.balanced),
		vulki.BindBuffer(2, resident.bounds),
		vulki.BindBuffer(3, resident.binarizer.params),
	)
	if err != nil {
		_ = bindings.histogram.Close()
		return gpuResidentInputBindings{}, fmt.Errorf("jabcode: bind resident GPU RGB balance input: %w", err)
	}
	resident.inputBindings[input] = bindings
	return bindings, nil
}

func (resident *gpuResidentBinarizer) preparedBindingsFor(
	input *vulki.Buffer,
) (gpuResidentPreparedBindings, error) {
	if bindings, ok := resident.preparedBindings[input]; ok {
		return bindings, nil
	}
	var bindings gpuResidentPreparedBindings
	var err error
	bindings.blocks, err = resident.blocksKernel.NewBindings(
		vulki.BindBuffer(0, input),
		vulki.BindBuffer(1, resident.binarizer.thresholds),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		return bindings, fmt.Errorf("jabcode: bind resident GPU block thresholds: %w", err)
	}
	bindings.classifier, err = resident.binarizer.classify.kernel.NewBindings(
		vulki.BindBuffer(0, input),
		vulki.BindBuffer(1, resident.binarizer.thresholds),
		vulki.BindBuffer(2, resident.binarizer.rawMasks),
		vulki.BindBuffer(3, resident.binarizer.params),
	)
	if err != nil {
		_ = bindings.blocks.Close()
		return gpuResidentPreparedBindings{}, fmt.Errorf("jabcode: bind resident GPU RGB classifier: %w", err)
	}
	resident.preparedBindings[input] = bindings
	return bindings, nil
}

func (resident *gpuResidentBinarizer) releasePreparedBindings(input *vulki.Buffer) error {
	if resident == nil || input == nil {
		return nil
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	bindings, ok := resident.preparedBindings[input]
	if !ok {
		return nil
	}
	delete(resident.preparedBindings, input)
	return errors.Join(bindings.classifier.Close(), bindings.blocks.Close())
}

// releaseInputBindings drops the cached histogram and balance binding sets of
// one input buffer. A route canvas about to replace its grown route buffer
// must release them first: the binding sets hold live references that keep
// the old buffer from closing.
func (resident *gpuResidentBinarizer) releaseInputBindings(input *vulki.Buffer) error {
	if resident == nil || input == nil {
		return nil
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	bindings, ok := resident.inputBindings[input]
	if !ok {
		return nil
	}
	delete(resident.inputBindings, input)
	return errors.Join(bindings.balance.Close(), bindings.histogram.Close())
}

func (resident *gpuResidentBinarizer) Binarize(
	input *vulki.Buffer,
	width, height int,
	blkThs []float32,
	printLevels bool,
	scanChannels uint32,
) ([3]*core.Bitmap, *finderPassRowHits, func() error, error) {
	var empty [3]*core.Bitmap
	if resident == nil {
		return empty, nil, nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() || resident.binarizer == nil {
		return empty, nil, nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	pixelCount, err := resident.validateBinarizationLocked(width, height, blkThs)
	if err != nil {
		return empty, nil, nil, err
	}
	if input == nil || input.Size() < uint64(pixelCount)*4 {
		return empty, nil, nil, fmt.Errorf("jabcode: resident GPU input buffer is too small")
	}
	bindings, err := resident.bindingsFor(input)
	if err != nil {
		return empty, nil, nil, err
	}
	params, blocksX, blocksY := gpuResidentBinarizerParams(width, height, blkThs, printLevels)
	preparedBindings, err := resident.preparedBindingsFor(resident.balanced)
	if err != nil {
		return empty, nil, nil, err
	}
	var zeroHistogram [gpuRGBHistogramBytes]byte
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: create resident GPU binarizer recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.histogram, 0, zeroHistogram[:]); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: clear resident GPU RGB histogram: %w", err)
	}
	if err := recorder.Update(resident.binarizer.params, 0, params[:]); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: update resident GPU binarizer parameters: %w", err)
	}
	pixelGroups := gpuCanvasWorkgroups(width, height)
	if err := recorder.Dispatch(resident.histogramKernel, bindings.histogram, pixelGroups); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: dispatch resident GPU RGB histogram: %w", err)
	}
	if err := recorder.Barrier(resident.histogram); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: synchronize resident GPU RGB histogram: %w", err)
	}
	if err := recorder.Dispatch(
		resident.boundsKernel,
		resident.boundsBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: dispatch resident GPU histogram bounds: %w", err)
	}
	if err := recorder.Barrier(resident.bounds); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: synchronize resident GPU histogram bounds: %w", err)
	}
	if err := recorder.Dispatch(resident.balanceKernel, bindings.balance, pixelGroups); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: dispatch resident GPU RGB balance: %w", err)
	}
	if err := recorder.Barrier(resident.balanced); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: synchronize resident GPU RGB balance: %w", err)
	}
	chainChannels, err := resident.recordPreparedBinarizationLocked(
		recorder, preparedBindings, width, height, blkThs, blocksX, blocksY, scanChannels, printLevels,
	)
	if err != nil {
		return empty, nil, nil, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: run resident GPU binarizer: %w", err)
	}
	chainChannels = resident.binarizer.downloadFinderScan(width, height, scanChannels, chainChannels, printLevels, resident.rowStride)
	channels, materialize := resident.lazyChannelsLocked(width, height)
	return channels, resident.scanHitsLocked(scanChannels, chainChannels), materialize, nil
}

func (resident *gpuResidentBinarizer) BinarizeBalanced(
	width, height int,
	blkThs []float32,
	printLevels bool,
	scanChannels uint32,
) ([3]*core.Bitmap, *finderPassRowHits, func() error, error) {
	if resident == nil {
		return [3]*core.Bitmap{}, nil, nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	return resident.BinarizePrepared(resident.balanced, width, height, blkThs, printLevels, scanChannels)
}

func (resident *gpuResidentBinarizer) BinarizePrepared(
	input *vulki.Buffer,
	width, height int,
	blkThs []float32,
	printLevels bool,
	scanChannels uint32,
) ([3]*core.Bitmap, *finderPassRowHits, func() error, error) {
	var empty [3]*core.Bitmap
	if resident == nil {
		return empty, nil, nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() || resident.binarizer == nil {
		return empty, nil, nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	pixelCount, err := resident.validateBinarizationLocked(width, height, blkThs)
	if err != nil {
		return empty, nil, nil, err
	}
	if input == nil || input.Size() < uint64(pixelCount)*4 {
		return empty, nil, nil, fmt.Errorf("jabcode: resident GPU prepared input buffer is too small")
	}
	params, blocksX, blocksY := gpuResidentBinarizerParams(width, height, blkThs, printLevels)
	preparedBindings, err := resident.preparedBindingsFor(input)
	if err != nil {
		return empty, nil, nil, err
	}
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: create resident GPU rebinarizer recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.binarizer.params, 0, params[:]); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: update resident GPU rebinarizer parameters: %w", err)
	}
	chainChannels, err := resident.recordPreparedBinarizationLocked(
		recorder, preparedBindings, width, height, blkThs, blocksX, blocksY, scanChannels, printLevels,
	)
	if err != nil {
		return empty, nil, nil, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, nil, nil, fmt.Errorf("jabcode: run resident GPU rebinarizer: %w", err)
	}
	chainChannels = resident.binarizer.downloadFinderScan(width, height, scanChannels, chainChannels, printLevels, resident.rowStride)
	channels, materialize := resident.lazyChannelsLocked(width, height)
	return channels, resident.scanHitsLocked(scanChannels, chainChannels), materialize, nil
}

// lazyChannelsLocked returns the pass's binarized channels as shape-only
// bitmaps plus the materializer that fetches and expands the packed words into
// them on first need. The masks stay on the device until then, and the packed
// host buffer is reused per pass, so the materializer is valid only until this
// binarizer's next pass and fails deterministically afterward.
func (resident *gpuResidentBinarizer) lazyChannelsLocked(
	width, height int,
) ([3]*core.Bitmap, func() error) {
	resident.generation++
	generation := resident.generation
	resident.masksOnHost = false
	var channels [3]*core.Bitmap
	for c := range channels {
		channels[c] = &core.Bitmap{Width: width, Height: height, Channels: 1}
	}
	resident.lazyChannels = channels
	materialize := func() error {
		resident.mu.Lock()
		defer resident.mu.Unlock()
		if resident.closed || resident.generation != generation {
			return fmt.Errorf("jabcode: resident GPU mask pass was superseded before materialization")
		}
		packedMasks, err := resident.packedMasksLocked(width, height)
		if err != nil {
			return err
		}
		shape := core.Bitmap{Width: width, Height: height}
		filled := unpackGPUBinarizerMasks(&shape, packedMasks)
		for c := range channels {
			channels[c].Pix = filled[c].Pix
		}
		return nil
	}
	return channels, materialize
}

// packedMasksLocked hands back this pass's packed mask words, fetching them
// from the device on the first request and reusing them afterward.
//
// The masks are what every host stage that reads d.Ch needs, and on a decode
// that keeps the finder work on the device no stage asks at all: the row scan,
// the directional sweeps, the sampler and the side-size walk all read the
// device buffer directly. Downloading them with the pass therefore moved the
// largest remaining transfer for nothing on the routes that matter, so the
// fetch waits here for a caller that genuinely reads pixels.
func (resident *gpuResidentBinarizer) packedMasksLocked(width, height int) ([]byte, error) {
	packedMasks := resident.binarizer.hostMasks[:((width*height+7)/8)*4]
	if resident.masksOnHost {
		return packedMasks, nil
	}
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create resident GPU mask downloader: %w", err)
	}
	defer recorder.Abort()
	phaseprobe.Count("download.packed_masks", len(packedMasks))
	if err := recorder.Download(resident.binarizer.packedMasks, 0, packedMasks); err != nil {
		return nil, fmt.Errorf("jabcode: record resident GPU binarizer mask download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: download resident GPU binarizer masks: %w", err)
	}
	resident.masksOnHost = true
	return packedMasks, nil
}

// snapshotChannels copies the current pass's downloaded packed mask words so
// their expansion can outlive the pass and its context lease. channels must
// be this binarizer's latest lazy pass; the returned materializer fills those
// bitmaps on first need without touching the binarizer again.
func (resident *gpuResidentBinarizer) snapshotChannels(channels [3]*core.Bitmap) (func() error, error) {
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.binarizer == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if channels != resident.lazyChannels {
		return nil, fmt.Errorf("jabcode: resident GPU mask snapshot requested for a superseded pass")
	}
	width, height := channels[0].Width, channels[0].Height
	size := uint64(((width*height + 7) / 8) * 4)
	// The words are copied device-side. That is what makes the snapshot free of
	// host traffic: a later pass may overwrite packedMasks, but this pass's
	// masks survive in a buffer the route's own lease keeps alive, and they are
	// fetched only if a reader actually turns up.
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU mask preserve recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Copy(
		resident.binarizer.preservedMasks, 0, resident.binarizer.packedMasks, 0, size,
	); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU mask preservation: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: preserve GPU masks: %w", err)
	}

	var packed []byte
	fetch := func() ([]byte, error) {
		if packed != nil {
			return packed, nil
		}
		out := make([]byte, size)
		phaseprobe.Count("download.preserved_masks", len(out))
		if err := resident.binarizer.preservedMasks.Download(out); err != nil {
			return nil, fmt.Errorf("jabcode: download preserved GPU masks: %w", err)
		}
		packed = out
		return packed, nil
	}
	for channel, bitmap := range channels {
		bitmap.SetPixelReader(func(x, y int) byte {
			words, err := fetch()
			if err != nil {
				return 0
			}
			pixel := y*width + x
			word := binary.LittleEndian.Uint32(words[(pixel/8)*4:])
			return b2byte(word>>uint((pixel%8)*3+channel)&1 != 0)
		})
	}
	return func() error {
		words, err := fetch()
		if err != nil {
			return err
		}
		shape := core.Bitmap{Width: width, Height: height}
		filled := unpackGPUBinarizerMasks(&shape, words)
		for c := range channels {
			channels[c].Pix = filled[c].Pix
			channels[c].SetPixelReader(nil)
		}
		return nil
	}, nil
}

func (resident *gpuResidentBinarizer) validateBinarizationLocked(
	width, height int,
	blkThs []float32,
) (int, error) {
	if width <= 0 || height <= 0 || width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return 0, fmt.Errorf(
			"jabcode: resident GPU image %dx%d exceeds configured maximum %dx%d",
			width, height, resident.binarizer.maxWidth, resident.binarizer.maxHeight,
		)
	}
	if blkThs != nil && len(blkThs) < 3 {
		return 0, fmt.Errorf("jabcode: resident GPU binarizer needs three fixed thresholds")
	}
	return width * height, nil
}

func (resident *gpuResidentBinarizer) recordPreparedBinarizationLocked(
	recorder *vulki.Recorder,
	bindings gpuResidentPreparedBindings,
	width, height int,
	blkThs []float32,
	blocksX, blocksY int,
	scanChannels uint32,
	printLevels bool,
) (uint32, error) {
	if blkThs == nil {
		if err := recorder.Dispatch(
			resident.blocksKernel,
			bindings.blocks,
			vulki.Workgroups{X: uint32(blocksX), Y: uint32(blocksY), Z: 1},
		); err != nil {
			return 0, fmt.Errorf("jabcode: dispatch resident GPU block thresholds: %w", err)
		}
		if err := recorder.Barrier(resident.binarizer.thresholds); err != nil {
			return 0, fmt.Errorf("jabcode: synchronize resident GPU block thresholds: %w", err)
		}
	}
	if err := resident.binarizer.recordComputeWithClassifier(
		recorder,
		bindings.classifier,
		width,
		height,
	); err != nil {
		return 0, err
	}
	var chainChannels uint32
	if scanChannels != 0 {
		var err error
		chainChannels, err = resident.binarizer.recordFinderScan(recorder, width, height, scanChannels, printLevels, resident.rowStride)
		if err != nil {
			return 0, err
		}
	}
	return chainChannels, nil
}

// SetRowStride records the consumer's finder walk spacing for the passes that
// follow. A detector sets it once, before its first binarize.
func (resident *gpuResidentBinarizer) SetRowStride(stride int) {
	if resident == nil {
		return
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	resident.rowStride = stride
}

// ScanDirection sweeps one probe direction over the masks the last binarize
// left resident. A nil result means the caller should sweep on the CPU.
func (resident *gpuResidentBinarizer) ScanDirection(
	width, height int,
	dir scanDirection,
	step, channel int,
) (finderDirSweep, error) {
	if resident == nil {
		return finderDirSweep{}, nil
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() || resident.binarizer == nil {
		return finderDirSweep{}, nil
	}
	return resident.binarizer.scanDirectionHits(width, height, dir, step, channel)
}

// scanHitsLocked parses the last recorded finder scan's downloaded records
// and chain outcomes, or returns nil when the pass did not scan.
// ScanDirectionBatch sweeps several directions over the still resident masks in
// one submission. A nil result means the caller should sweep them one at a time.
func (resident *gpuResidentBinarizer) ScanDirectionBatch(
	width, height int,
	dirs []scanDirection,
	step, channel int,
) ([]finderDirSweep, error) {
	if resident == nil || len(dirs) == 0 {
		return nil, nil
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() || resident.binarizer == nil {
		return nil, nil
	}
	return resident.binarizer.scanDirectionBatchHits(width, height, dirs, step, channel)
}

func (resident *gpuResidentBinarizer) scanHitsLocked(scanChannels, chainChannels uint32) *finderPassRowHits {
	if scanChannels == 0 {
		return nil
	}
	// A pass whose fold covers every scanned channel never downloaded its raw
	// records, so its hits are restored from the summary and the compacted list.
	if covered := resident.binarizer.rowSummaryValid; covered&scanChannels == scanChannels {
		return parseFinderRowSummary(
			resident.binarizer.hostRowSummary,
			resident.binarizer.hostRowCompacted,
			scanChannels,
			covered,
		)
	}
	chainOutcomes := resident.binarizer.hostChainOutcomes
	if chainChannels == 0 {
		chainOutcomes = nil
	}
	return parseFinderScanRecords(
		resident.binarizer.hostScanRecords,
		chainOutcomes,
		scanChannels,
		chainChannels,
	)
}

func (resident *gpuResidentBinarizer) DownloadBalanced(
	width, height int,
) (*core.Bitmap, error) {
	if resident == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	return resident.DownloadPrepared(resident.balanced, width, height)
}

func (resident *gpuResidentBinarizer) DownloadPrepared(
	input *vulki.Buffer,
	width, height int,
) (*core.Bitmap, error) {
	if resident == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || input == nil || resident.binarizer == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if width <= 0 || height <= 0 || width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return nil, fmt.Errorf(
			"jabcode: resident GPU image %dx%d exceeds configured maximum %dx%d",
			width, height, resident.binarizer.maxWidth, resident.binarizer.maxHeight,
		)
	}
	bm := core.NewBitmap(width, height, 4)
	if input.Size() < uint64(len(bm.Pix)) {
		return nil, fmt.Errorf("jabcode: resident GPU prepared input buffer is too small")
	}
	phaseprobe.Count("download.prepared_image", len(bm.Pix))
	if err := input.Download(bm.Pix); err != nil {
		return nil, fmt.Errorf("jabcode: download resident GPU prepared image: %w", err)
	}
	return bm, nil
}

func gpuResidentBinarizerParams(
	width, height int,
	blkThs []float32,
	printLevels bool,
) (params [gpuBinarizerParamsSize]byte, blocksX, blocksY int) {
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	flags := uint32(0)
	if blkThs == nil {
		blockSize := capInt(min(width, height)/binThresholdDivisor, binMinBlock, binMaxBlock)
		blocksX = (width + blockSize - 1) / blockSize
		blocksY = (height + blockSize - 1) / blockSize
		binary.LittleEndian.PutUint32(params[8:], uint32(blockSize))
		binary.LittleEndian.PutUint32(params[12:], uint32(blocksX))
		binary.LittleEndian.PutUint32(params[16:], uint32(blocksY))
	} else {
		flags |= 1
		blocksX, blocksY = 1, 1
		binary.LittleEndian.PutUint32(params[8:], 1)
		binary.LittleEndian.PutUint32(params[12:], 1)
		binary.LittleEndian.PutUint32(params[16:], 1)
		binary.LittleEndian.PutUint32(params[24:], math.Float32bits(blkThs[0]))
		binary.LittleEndian.PutUint32(params[28:], math.Float32bits(blkThs[1]))
		binary.LittleEndian.PutUint32(params[32:], math.Float32bits(blkThs[2]))
	}
	if printLevels {
		flags |= 2
	}
	binary.LittleEndian.PutUint32(params[20:], flags)
	return params, blocksX, blocksY
}

func (resident *gpuResidentBinarizer) Close() error {
	if resident == nil {
		return nil
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed {
		return nil
	}
	resident.closed = true
	return resident.closeResources()
}

func (resident *gpuResidentBinarizer) closeResources() error {
	var closeErrors []error
	for input, bindings := range resident.preparedBindings {
		closeErrors = append(closeErrors, bindings.classifier.Close(), bindings.blocks.Close())
		delete(resident.preparedBindings, input)
	}
	for input, bindings := range resident.inputBindings {
		closeErrors = append(closeErrors, bindings.balance.Close(), bindings.histogram.Close())
		delete(resident.inputBindings, input)
	}
	for _, bindings := range []*vulki.BindingSet{
		resident.boundsBindings, resident.sampleBindings, resident.moduleCountBindings,
		resident.alignBindings, resident.ldpcBindings,
		resident.payloadMapBindings, resident.payloadPermuteBindings,
		resident.payloadBitsBindings, resident.metadataPart1Bindings,
		resident.metadataPaletteBindings, resident.metadataPart2Bindings,
		resident.metadataFinishBindings,
		resident.foldBindings,
		resident.sortBindings,
	} {
		if bindings != nil {
			closeErrors = append(closeErrors, bindings.Close())
		}
	}
	resident.boundsBindings = nil
	resident.sampleBindings = nil
	resident.moduleCountBindings = nil
	resident.alignBindings = nil
	resident.ldpcBindings = nil
	resident.payloadMapBindings = nil
	resident.payloadPermuteBindings = nil
	resident.payloadBitsBindings = nil
	resident.metadataPart1Bindings = nil
	resident.metadataPaletteBindings = nil
	resident.metadataPart2Bindings = nil
	resident.metadataFinishBindings = nil
	resident.foldBindings = nil
	resident.sortBindings = nil
	// The kernels belong to the shared per-device set; this instance only
	// drops its references.
	resident.metadataPart1Kernel = nil
	resident.metadataPaletteKernel = nil
	resident.metadataPart2Kernel = nil
	resident.metadataFinishKernel = nil
	resident.alignKernel = nil
	resident.ldpcKernel = nil
	resident.payloadMapKernel = nil
	resident.payloadPermuteKernel = nil
	resident.payloadBitsKernel = nil
	resident.moduleCountKernel = nil
	resident.sampleKernel = nil
	resident.blocksKernel = nil
	resident.balanceKernel = nil
	resident.boundsKernel = nil
	resident.histogramKernel = nil
	// The binarizer goes first: its chain binding set references the balanced
	// image, and a buffer cannot be released while a binding still holds it.
	if resident.binarizer != nil {
		closeErrors = append(closeErrors, resident.binarizer.Close())
		resident.binarizer = nil
	}
	for _, buffer := range []*vulki.Buffer{
		resident.balanced, resident.bounds, resident.histogram,
		resident.sampleResult, resident.sampleParams,
		resident.moduleCountResult, resident.moduleCountParams,
		resident.alignCells, resident.alignParams, resident.alignTiles,
		resident.ldpcRows, resident.ldpcBits, resident.ldpcParams, resident.ldpcNet,
		resident.payloadParams, resident.payloadMap, resident.payloadPermutation,
		resident.metadataParams, resident.metadataRecord,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	resident.balanced = nil
	resident.bounds = nil
	resident.histogram = nil
	resident.sampleResult = nil
	resident.sampleParams = nil
	resident.moduleCountResult = nil
	resident.moduleCountParams = nil
	resident.alignCells = nil
	resident.alignParams = nil
	resident.alignTiles = nil
	resident.ldpcRows = nil
	resident.ldpcBits = nil
	resident.ldpcParams = nil
	resident.ldpcNet = nil
	resident.payloadParams = nil
	resident.payloadMap = nil
	resident.payloadPermutation = nil
	resident.sampledGrid = nil
	resident.permutationLength = 0
	if resident.ownsKernels {
		closeErrors = append(closeErrors, resident.kernels.Close())
	}
	resident.kernels = nil
	resident.device = nil
	return errors.Join(closeErrors...)
}
