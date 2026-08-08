//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/phaseprobe"
	"github.com/srlehn/jabcode/internal/spec"

	"github.com/srlehn/vulki"
)

//go:embed shaders/finder_row_scan.wgsl
var finderRowScanWGSL string

//go:embed shaders/finder_line_scan.wgsl
var finderLineScanWGSL string

//go:embed shaders/finder_scan_params.wgsl
var finderScanParamsWGSL string

//go:embed shaders/finder_scan_mask_packed.wgsl
var finderScanMaskPackedWGSL string

//go:embed shaders/finder_scan_mask_plane.wgsl
var finderScanMaskPlaneWGSL string

//go:embed shaders/finder_scan_geometry.wgsl
var finderScanGeometryWGSL string

//go:embed shaders/finder_runs_hillis.wgsl
var finderRunsHillisWGSL string

//go:embed shaders/finder_runs_subgroup.wgsl
var finderRunsSubgroupWGSL string

//go:embed shaders/finder_cross_check.wgsl
var finderCrossCheckWGSL string

//go:embed shaders/finder_windows_common.wgsl
var finderWindowsCommonWGSL string

//go:embed shaders/finder_windows_ballot.wgsl
var finderWindowsBallotWGSL string

//go:embed shaders/finder_windows_scan.wgsl
var finderWindowsScanWGSL string

//go:embed shaders/enable_subgroups.wgsl
var enableSubgroupsWGSL string

//go:embed shaders/subgroup_probe.wgsl
var subgroupProbeWGSL string

//go:embed shaders/finder_chain_bindings.wgsl
var finderChainBindingsWGSL string

//go:embed shaders/finder_chain_prelude.wgsl
var finderChainPreludeWGSL string

//go:embed shaders/finder_chain_row.wgsl
var finderChainRowWGSL string

//go:embed shaders/finder_chain_current.wgsl
var finderChainCurrentWGSL string

//go:embed shaders/finder_chain_directional_bindings.wgsl
var finderChainDirectionalBindingsWGSL string

//go:embed shaders/finder_chain_directional.wgsl
var finderChainDirectionalWGSL string

//go:embed shaders/finder_chain_directional_current.wgsl
var finderChainDirectionalCurrentWGSL string

//go:embed shaders/finder_dispatch_args.wgsl
var finderDispatchArgsWGSL string

const (
	gpuBinarizerWorkgroupWidth  = 8
	gpuBinarizerWorkgroupHeight = 8
	gpuPackWorkgroupSize        = 64
	gpuBinarizerParamsSize      = 48
	gpuThresholdCellSize        = 32

	// gpuFinderScanCapacity is the initial raw-hit record capacity of one
	// scan pass. Most passes seed hundreds to a few thousand raw hits, but a
	// noisy full-resolution adverse capture measures up to about 0.02 records
	// per pixel; a pass that overflows reads the true count back and retries
	// once over the still resident masks with grown buffers, so the cap risks
	// throughput, never correctness.
	gpuFinderScanCapacity    = 65536
	gpuFinderScanParamsSize  = 16
	gpuFinderScanBufferBytes = gpuFinderScanHeaderBytes +
		gpuFinderScanCapacity*gpuFinderScanRecordWords*4
	gpuFinderScanWorkgroupSize = 64

	gpuFinderChainBufferBytes   = gpuFinderScanCapacity * gpuFinderChainOutcomeWords * 4
	gpuFinderChainParamsSize    = 32
	gpuFinderChainWorkgroupSize = 64

	// Bits 8 and up of the chain flags carry the consumer's row stride.
	gpuChainFlagStrideShift = 8
)

// gpuFinderScanBufferSize returns the record buffer bytes for a capacity.
func gpuFinderScanBufferSize(capacity int) int {
	return gpuFinderScanHeaderBytes + capacity*gpuFinderScanRecordWords*4
}

// gpuFinderChainBufferSize returns the chain outcome buffer bytes for a
// capacity.
func gpuFinderChainBufferSize(capacity int) int {
	return capacity * gpuFinderChainOutcomeWords * 4
}

// gpuFinderScanGrowthBytes returns the retained device bytes an overflow
// growth from the old to the new record capacity adds: the record buffer
// plus the chain outcome buffer.
func gpuFinderScanGrowthBytes(oldCapacity, newCapacity int) uint64 {
	return uint64(gpuFinderScanBufferSize(newCapacity)-gpuFinderScanBufferSize(oldCapacity)) +
		uint64(gpuFinderChainBufferSize(newCapacity)-gpuFinderChainBufferSize(oldCapacity))
}

// gpuFinderScanMaxCapacity bounds overflow growth to one record per eight
// pixels. The run-length machine consumes at least seven pixels per raw hit,
// so a canvas beyond this bound is pathological noise; its pass keeps the
// bit-identical CPU row walk instead of growing further.
func gpuFinderScanMaxCapacity(width, height int) int {
	return max(gpuFinderScanCapacity, width*height/8)
}

type gpuBinarizerStage struct {
	kernel   *vulki.Kernel
	bindings *vulki.BindingSet
}

// gpuBinarizer is a measurement surface for the fused RGB classification and
// binary-filter chain. Its buffers and bindings are reused across calls; the
// compute kernels come from a shared per-device set so concurrent route
// contexts do not recompile WGSL. It is deliberately internal until parity
// and transfer measurements establish a useful integration boundary.
type gpuBinarizer struct {
	mu sync.Mutex

	device      *vulki.Device
	kernels     *gpuDecodeKernels
	ownsKernels bool
	ownsDevice  bool
	closed      bool
	maxWidth    int
	maxHeight   int

	input *vulki.Buffer
	// colorSource is the balanced image the directional chain samples for its
	// colour-signal verdict. It is nil for binarizers whose owner keeps no
	// balanced image, and then the host answers that question instead.
	colorSource *vulki.Buffer
	thresholds  *vulki.Buffer
	rawMasks    *vulki.Buffer
	finalMasks  *vulki.Buffer
	packedMasks *vulki.Buffer
	params      *vulki.Buffer
	hostMasks   []byte

	// The directional sweep's own record, counter and parameter buffers. The
	// row scan's scanRecords carries a different layout and cannot be shared.
	// Created on the first directional pass rather than at initialization: the
	// record buffer alone is 8 MB, and a pass whose row walk settles never
	// needs any of them. How many reads that is has never been counted.
	dirRecords       *vulki.Buffer
	dirCounters      *vulki.Buffer
	dirParams        *vulki.Buffer
	dirBindings      *vulki.BindingSet
	dirChainOutcomes *vulki.Buffer
	// dirSummary holds the counters and module histogram the chain folds every
	// hit into, so a direction reads back one small block instead of one record
	// per hit.
	dirSummary       *vulki.Buffer
	dirChainParams   *vulki.Buffer
	dirChainBindings *vulki.BindingSet
	// dirChainBSIBindings is the BSI-era family's view of the same chain state.
	// Nil in a build that compiles no BSI family.
	dirChainBSIBindings *vulki.BindingSet
	// dirArgs is the chain's indirect dispatch command, written on the device
	// from the scan's own counter so that count never comes back to the host.
	dirArgs         *vulki.Buffer
	dirArgsBindings *vulki.BindingSet

	scanRecords     *vulki.Buffer
	scanParams      *vulki.Buffer
	hostScanRecords []byte
	scanCapacity    int

	// onRetainedAllocation reports lazy allocation and retained device-buffer
	// growth beyond the context's admitted base, in bytes. The route context
	// pool charges it to its memory budget. Called under this binarizer's mutex,
	// so the hook must not wait on locks this binarizer's callers hold.
	onRetainedAllocation func(delta uint64)

	chainOutcomes     *vulki.Buffer
	chainParams       *vulki.Buffer
	hostChainOutcomes []byte

	// rowSummary folds every hit's counters and module size per scan channel,
	// and rowCompacted holds only the candidates the consumer can act on, so a
	// pass reads a short list instead of every raw record and every outcome.
	preservedMasks   *vulki.Buffer
	dirBatchSummary  *vulki.Buffer
	dirBatchOutcomes *vulki.Buffer
	rowSummary       *vulki.Buffer
	rowCompacted     *vulki.Buffer

	// seedHistogram is the module-size distribution both the row chain and the
	// directional chain add into, across every scan of one locate. Its only
	// consumer reads it once, so it accumulates here rather than riding back in
	// each summary.
	seedHistogram    *vulki.Buffer
	hostRowSummary   []byte
	hostRowCompacted []byte
	// rowSummaryValid marks the channels whose downloaded fold is complete, so
	// a consumer knows when the raw records were never read.
	rowSummaryValid        uint32
	chainStageErr          error
	directionalPrintLevels bool

	// scanOnly keeps the optional device replay tiers - the per-hit
	// cross-check chains here and the preparer's resident pitch fold - off
	// the device, leaving the bit-identical CPU twins to classify the row
	// hits the scan seeds. It exists to exercise those twins deterministically;
	// ordinary contexts replay on the device.
	//
	// Replay used to be the exception, on a measurement that turned out to
	// have the wrong sign. What actually drives the cost is how many candidate
	// hits the row scan produces, and scan-only pays for every rejected one on
	// the CPU: small modules make the scan fire freely, so an in-process arm
	// comparison found scan-only nearly four times slower on a 3-pixel-module
	// symbol and around a quarter slower on full-resolution captures. Replay
	// loses only on a small symbol with large modules, where it costs under a
	// millisecond on a read that takes three.
	scanOnly bool

	classify gpuBinarizerStage
	filter   gpuBinarizerStage
	pack     gpuBinarizerStage
	scan     gpuBinarizerStage
	chain    gpuBinarizerStage
	chainBSI gpuBinarizerStage
}

func newGPUBinarizer(maxWidth, maxHeight int) (*gpuBinarizer, error) {
	device, err := vulki.Open()
	if err != nil {
		return nil, fmt.Errorf("jabcode: open GPU binarizer device: %w", err)
	}
	kernels := newGPUDecodeKernels(device)
	binarizer, err := newGPUBinarizerPipelineWithDevice(device, kernels, maxWidth, maxHeight, true)
	if err == nil {
		// A standalone binarizer compiles its chain kernels up front; only
		// the shared decode workspace warms them in the background.
		err = kernels.compileFinderChains()
	}
	if err != nil {
		if binarizer != nil {
			_ = binarizer.Close()
		}
		_ = kernels.Close()
		_ = device.Close()
		return nil, err
	}
	binarizer.ownsDevice = true
	binarizer.ownsKernels = true
	return binarizer, nil
}

func newGPUBinarizerPipelineWithDevice(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	maxWidth, maxHeight int,
	hostInput bool,
) (*gpuBinarizer, error) {
	if device == nil || device.Closed() {
		return nil, fmt.Errorf("jabcode: GPU binarizer device is closed")
	}
	if maxWidth <= 0 || maxHeight <= 0 {
		return nil, fmt.Errorf("jabcode: GPU binarizer dimensions must be positive")
	}
	area := uint64(maxWidth) * uint64(maxHeight)
	if area > uint64(math.MaxUint32) || area > uint64(math.MaxInt) {
		return nil, fmt.Errorf("jabcode: GPU binarizer image area exceeds shader limits")
	}

	b := &gpuBinarizer{device: device, kernels: kernels, maxWidth: maxWidth, maxHeight: maxHeight}
	if err := b.initialize(hostInput); err != nil {
		_ = b.closeResources()
		return nil, err
	}
	return b, nil
}

func (b *gpuBinarizer) initialize(hostInput bool) error {
	area := uint64(b.maxWidth) * uint64(b.maxHeight)
	maxBlocksX := (b.maxWidth + binMinBlock - 1) / binMinBlock
	maxBlocksY := (b.maxHeight + binMinBlock - 1) / binMinBlock
	thresholdBytes := uint64(maxBlocksX) * uint64(maxBlocksY) * gpuThresholdCellSize

	var err error
	if hostInput {
		b.input, err = b.device.NewBuffer(area * 4)
		if err != nil {
			return fmt.Errorf("jabcode: allocate GPU input: %w", err)
		}
	}
	b.thresholds, err = b.device.NewBuffer(thresholdBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU thresholds: %w", err)
	}
	b.rawMasks, err = b.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU raw masks: %w", err)
	}
	b.finalMasks, err = b.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU final masks: %w", err)
	}
	packedWords := (area + 7) / 8
	b.packedMasks, err = b.device.NewBuffer(packedWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU packed masks: %w", err)
	}
	// preservedMasks holds one located pass's packed words on the device so a
	// consumer that turns up later still reads that pass's masks after a
	// following pass overwrote packedMasks. Preserving on the device rather
	// than on the host is what keeps a located pass from paying a transfer for
	// readers it usually does not have.
	b.preservedMasks, err = b.device.NewBuffer(packedWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU preserved masks: %w", err)
	}
	b.hostMasks = make([]byte, packedWords*4)
	b.params, err = b.device.NewBuffer(gpuBinarizerParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU parameters: %w", err)
	}

	b.classify.kernel, err = b.kernels.classifyRGB()
	if err != nil {
		return err
	}
	if hostInput {
		b.classify.bindings, err = b.classify.kernel.NewBindings(
			vulki.BindBuffer(0, b.input),
			vulki.BindBuffer(1, b.thresholds),
			vulki.BindBuffer(2, b.rawMasks),
			vulki.BindBuffer(3, b.params),
		)
		if err != nil {
			return fmt.Errorf("jabcode: bind GPU RGB classifier: %w", err)
		}
	}
	b.filter, err = b.newStage(
		b.kernels.filterBinary,
		vulki.BindBuffer(0, b.rawMasks),
		vulki.BindBuffer(1, b.finalMasks),
		vulki.BindBuffer(2, b.params),
	)
	if err != nil {
		return fmt.Errorf("jabcode: create GPU binary filter: %w", err)
	}
	b.pack, err = b.newStage(
		b.kernels.packMasks,
		vulki.BindBuffer(0, b.finalMasks),
		vulki.BindBuffer(1, b.packedMasks),
		vulki.BindBuffer(2, b.params),
	)
	if err != nil {
		return fmt.Errorf("jabcode: create GPU mask packer: %w", err)
	}
	b.scanCapacity = gpuFinderScanCapacity
	b.scanRecords, err = b.device.NewBuffer(gpuFinderScanBufferBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder scan records: %w", err)
	}
	b.hostScanRecords = make([]byte, gpuFinderScanBufferBytes)
	b.scanParams, err = b.device.NewBuffer(gpuFinderScanParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder scan parameters: %w", err)
	}
	b.scan, err = b.newStage(
		b.kernels.finderRowScan,
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, b.scanRecords),
		vulki.BindBuffer(2, b.scanParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: create GPU finder row scan: %w", err)
	}
	b.chainOutcomes, err = b.device.NewBuffer(gpuFinderChainBufferBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder chain outcomes: %w", err)
	}
	b.hostChainOutcomes = make([]byte, gpuFinderChainBufferBytes)
	b.chainParams, err = b.device.NewBuffer(gpuFinderChainParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder chain parameters: %w", err)
	}
	b.rowSummary, err = b.device.NewBuffer(gpuRowSummaryBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder chain summary: %w", err)
	}
	b.hostRowSummary = make([]byte, gpuRowSummaryBytes)
	b.seedHistogram, err = b.device.NewBuffer(moduleSeedsBuckets * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU seed histogram: %w", err)
	}
	b.rowCompacted, err = b.device.NewBuffer(gpuRowCompactBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder chain compacted list: %w", err)
	}
	b.hostRowCompacted = make([]byte, gpuRowCompactBytes)
	// The chain stages bind lazily in chainChannels once the shared kernels
	// finish their background compilation.
	return nil
}

// chainChannels reports which requested channels get device chain outcomes
// this pass, binding the chain stages on first use after the shared kernels
// finish compiling. A scan-only pass keeps the bit-identical CPU per-hit
// chain in the consumer: that is what scanOnly selects permanently (see the
// field), and it is also the transitional mode everywhere else until the
// chain kernels finish compiling, which is what makes replay safe to enable
// by default. A failed stage bind latches chain use off rather than retrying
// every pass.
func (b *gpuBinarizer) chainChannels(channelMask uint32) uint32 {
	if b.scanOnly || channelMask == 0 || b.chainStageErr != nil ||
		!b.kernels.finderChainsReady() {
		return 0
	}
	if b.chain.bindings == nil {
		// A binarizer without a balanced image binds the packed masks in the
		// colour slot: the kernel never reads it, because the parameter flag
		// that enables the colour stage stays clear, and Vulkan still needs
		// every declared binding filled.
		colorSource := b.colorSource
		if colorSource == nil {
			colorSource = b.packedMasks
		}
		stage, err := b.newStage(
			b.kernels.finderChain,
			vulki.BindBuffer(0, b.packedMasks),
			vulki.BindBuffer(1, b.scanRecords),
			vulki.BindBuffer(2, b.chainOutcomes),
			vulki.BindBuffer(3, b.chainParams),
			vulki.BindBuffer(4, colorSource),
			vulki.BindBuffer(5, b.rowSummary),
			vulki.BindBuffer(6, b.rowCompacted),
			vulki.BindBuffer(7, b.seedHistogram),
		)
		if err != nil {
			b.chainStageErr = err
			return 0
		}
		b.chain = stage
		if bsiFamilyFinderEnabled {
			stageBSI, err := b.newStage(
				b.kernels.finderChainBSI,
				vulki.BindBuffer(0, b.packedMasks),
				vulki.BindBuffer(1, b.scanRecords),
				vulki.BindBuffer(2, b.chainOutcomes),
				vulki.BindBuffer(3, b.chainParams),
				vulki.BindBuffer(4, colorSource),
				vulki.BindBuffer(5, b.rowSummary),
				vulki.BindBuffer(6, b.rowCompacted),
				vulki.BindBuffer(7, b.seedHistogram),
			)
			if err != nil {
				b.chainStageErr = err
				return 0
			}
			b.chainBSI = stageBSI
		}
	}
	available := channelMask & (1 << 1)
	if bsiFamilyFinderEnabled {
		available |= channelMask & (1 << 0)
	}
	return available
}

// recordFinderScan appends the packed-mask row scan for the requested channel
// mask to a recording whose mask packer already ran and chains each available
// family's per-hit cross-check kernel over the raw records in the same
// submission. It returns the channel mask whose chain outcomes are
// device-computed this pass; after the submission completes the caller reads
// the buffers back with downloadFinderScan, sized to the actual record count.
func (b *gpuBinarizer) recordFinderScan(
	recorder *vulki.Recorder,
	width, height int,
	channelMask uint32,
	printLevels bool,
	rowStride int,
) (uint32, error) {
	b.directionalPrintLevels = printLevels
	var params [gpuFinderScanParamsSize]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	binary.LittleEndian.PutUint32(params[8:], channelMask)
	binary.LittleEndian.PutUint32(params[12:], uint32(b.scanCapacity))
	if err := recorder.Update(b.scanParams, 0, params[:]); err != nil {
		return 0, fmt.Errorf("jabcode: update GPU finder scan parameters: %w", err)
	}
	chainChannels := b.chainChannels(channelMask)
	if chainChannels != 0 {
		chainParams := gpuFinderChainParams(width, height, b.scanCapacity, printLevels)
		flags := binary.LittleEndian.Uint32(chainParams[12:])
		if b.colorSource != nil {
			flags |= gpuFinderChainFlagColorSource
		}
		// The consumer walks rows at this stride, so the fold has to skip the
		// same rows or its counters describe a scan nobody ran.
		flags |= uint32(max(rowStride, 1)) << gpuChainFlagStrideShift
		binary.LittleEndian.PutUint32(chainParams[12:], flags)
		binary.LittleEndian.PutUint32(chainParams[28:], uint32(gpuRowCompactCapacity))
		if err := recorder.Update(b.chainParams, 0, chainParams[:]); err != nil {
			return 0, fmt.Errorf("jabcode: update GPU finder chain parameters: %w", err)
		}
		// Every counter in the fold accumulates, so the block starts clear for
		// this pass rather than carrying the last one's totals.
		if err := recorder.Update(b.rowSummary, 0, make([]byte, gpuRowSummaryBytes)); err != nil {
			return 0, fmt.Errorf("jabcode: clear GPU finder chain summary: %w", err)
		}
		if err := recorder.Barrier(b.rowSummary); err != nil {
			return 0, fmt.Errorf("jabcode: synchronize GPU finder chain summary reset: %w", err)
		}
	}
	var header [gpuFinderScanHeaderBytes]byte
	if err := recorder.Update(b.scanRecords, 0, header[:]); err != nil {
		return 0, fmt.Errorf("jabcode: clear GPU finder scan counter: %w", err)
	}
	if err := recorder.Barrier(b.packedMasks); err != nil {
		return 0, fmt.Errorf("jabcode: synchronize GPU packed masks for the finder scan: %w", err)
	}
	groups := vulki.Workgroups{
		X: uint32((height + gpuFinderScanWorkgroupSize - 1) / gpuFinderScanWorkgroupSize),
		Y: 1,
		Z: 1,
	}
	if err := recorder.Dispatch(b.scan.kernel, b.scan.bindings, groups); err != nil {
		return 0, fmt.Errorf("jabcode: dispatch GPU finder row scan: %w", err)
	}
	if err := recorder.Barrier(b.scanRecords); err != nil {
		return 0, fmt.Errorf("jabcode: synchronize GPU finder scan records: %w", err)
	}
	if chainChannels != 0 {
		chainGroups := vulki.Workgroups{
			X: uint32((b.scanCapacity + gpuFinderChainWorkgroupSize - 1) / gpuFinderChainWorkgroupSize),
			Y: 1,
			Z: 1,
		}
		// Each family kernel writes only its own channel's outcome slots, so
		// the dispatches are independent.
		if chainChannels&(1<<1) != 0 {
			if err := recorder.Dispatch(b.chain.kernel, b.chain.bindings, chainGroups); err != nil {
				return 0, fmt.Errorf("jabcode: dispatch GPU finder chain: %w", err)
			}
		}
		if chainChannels&(1<<0) != 0 {
			if err := recorder.Dispatch(b.chainBSI.kernel, b.chainBSI.bindings, chainGroups); err != nil {
				return 0, fmt.Errorf("jabcode: dispatch GPU BSI finder chain: %w", err)
			}
		}
		if err := recorder.Barrier(b.chainOutcomes); err != nil {
			return 0, fmt.Errorf("jabcode: synchronize GPU finder chain outcomes: %w", err)
		}
	}
	// The header is only sixteen bytes and its count sizes the later record
	// readback. Keeping it in this submission avoids creating a transient
	// command pool solely to discover how much output the dispatch produced.
	if err := recorder.Download(b.scanRecords, 0, b.hostScanRecords[:gpuFinderScanHeaderBytes]); err != nil {
		return 0, fmt.Errorf("jabcode: record GPU finder scan counter download: %w", err)
	}
	return chainChannels, nil
}

// downloadFinderScan reads the submitted pass's scan records and chain
// outcomes back, sized to the actual record count instead of the buffer
// capacity. An overflowed scan retries once over the still resident packed
// masks with buffers grown to the reported count. Every failure path leaves
// the host records parsing as overflowed, so the consumer runs the
// bit-identical CPU row walk and the readback risks throughput, never
// correctness.
func (b *gpuBinarizer) downloadFinderScan(
	width, height int,
	channelMask, chainChannels uint32,
	printLevels bool,
	rowStride int,
) uint32 {
	if channelMask == 0 {
		return chainChannels
	}
	poison := func() {
		binary.LittleEndian.PutUint32(b.hostScanRecords, math.MaxUint32)
	}
	// The fold belongs to one pass; a later pass that never reads it must not
	// inherit the last one's coverage.
	b.rowSummaryValid = 0
	count := b.scanRecordCount()
	if count > b.scanCapacity {
		if count > gpuFinderScanMaxCapacity(width, height) {
			return chainChannels
		}
		if err := b.growFinderScan(count); err != nil {
			return chainChannels
		}
		rescanned, err := b.rescanFinderScan(width, height, channelMask, printLevels, rowStride)
		if err != nil {
			return chainChannels
		}
		chainChannels = rescanned
		count = b.scanRecordCount()
		if count > b.scanCapacity {
			return chainChannels
		}
	}
	if count == 0 {
		return chainChannels
	}
	// A chained channel comes back as a summary and a short candidate list, so
	// its raw records and its per-hit outcomes never cross the bus. Only a
	// channel the chain did not cover still needs its records, and a chained
	// channel whose candidates outgrew their region falls back to the same
	// reading rather than acting on a prefix.
	if chainChannels != 0 {
		summarized, err := b.downloadRowSummary(chainChannels)
		if err == nil && summarized&channelMask == channelMask {
			return chainChannels
		}
		if err != nil {
			b.rowSummaryValid = 0
		}
	}
	prefix := gpuFinderScanHeaderBytes + count*gpuFinderScanRecordWords*4
	recorder, err := b.device.NewRecorder()
	if err != nil {
		poison()
		return chainChannels
	}
	defer recorder.Abort()
	phaseprobe.Count("download.row_scan_records", prefix)
	if err := recorder.Download(b.scanRecords, 0, b.hostScanRecords[:prefix]); err != nil {
		poison()
		return chainChannels
	}
	if chainChannels != 0 {
		phaseprobe.Count("download.row_chain_outcomes", count*gpuFinderChainOutcomeWords*4)
		if err := recorder.Download(b.chainOutcomes, 0, b.hostChainOutcomes[:count*gpuFinderChainOutcomeWords*4]); err != nil {
			// Neither recorded download has run yet, so poison the raw
			// records too and let the consumer repeat the row walk.
			poison()
			return 0
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		poison()
		return 0
	}
	return chainChannels
}

// downloadRowSummary reads the fold and reports which channels it fully covers.
// A channel whose region overflowed is left out, so the caller reads that pass's
// raw records instead.
//
// The compacted candidates stay where the chain wrote them. A pass that folds on
// the device never needs them on this side at all, and the counters that decide
// whether it can are in the summary, so fetching them here would pay for the
// common case in order to serve the rare one.
func (b *gpuBinarizer) downloadRowSummary(chainChannels uint32) (uint32, error) {
	recorder, err := b.device.NewRecorder()
	if err != nil {
		return 0, err
	}
	defer recorder.Abort()
	phaseprobe.Count("download.row_summary", len(b.hostRowSummary))
	if err := recorder.Download(b.rowSummary, 0, b.hostRowSummary); err != nil {
		return 0, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return 0, err
	}
	// Only a channel whose chain actually ran has a fold to read; a scanned
	// channel without one still needs its raw records, and claiming coverage
	// for it would hand the consumer an empty list instead.
	covered := chainChannels
	for channel := range gpuRowSummaryChannels {
		if covered&(1<<channel) == 0 {
			continue
		}
		block := channel * gpuRowSummaryWords
		overflow := binary.LittleEndian.Uint32(
			b.hostRowSummary[(block+gpuRowSummaryOverflow)*4:],
		)
		if overflow != 0 {
			covered &^= 1 << channel
		}
	}
	b.rowSummaryValid = covered
	return covered, nil
}

// downloadRowCompacted fetches one channel's compacted candidates for a host arm
// that has to read them. Each channel owns a fixed region, so the used prefix is
// read on its own rather than as one span reaching across the gaps between the
// regions, which would be the whole buffer whatever the counts.
func (b *gpuBinarizer) downloadRowCompacted(channel, count int) bool {
	if b == nil || b.device == nil || b.rowCompacted == nil ||
		channel < 0 || channel >= gpuRowSummaryChannels ||
		count <= 0 || count > gpuRowCompactCapacity ||
		b.rowSummaryValid&(1<<channel) == 0 {
		return false
	}
	recorder, err := b.device.NewRecorder()
	if err != nil {
		return false
	}
	defer recorder.Abort()
	start := channel * gpuRowCompactCapacity * gpuRowCompactWords * 4
	length := count * gpuRowCompactWords * 4
	phaseprobe.Count("download.row_compacted", length)
	if err := recorder.Download(
		b.rowCompacted, uint64(start), b.hostRowCompacted[start:start+length],
	); err != nil {
		return false
	}
	return recorder.SubmitAndWait() == nil
}

// scanRecordCount reads the record counter of the last downloaded finder
// scan. The scan kernel counts every hit even past the buffer capacity, so a
// value above the capacity is the exact size an overflow retry needs.
func (b *gpuBinarizer) scanRecordCount() int {
	return int(binary.LittleEndian.Uint32(b.hostScanRecords))
}

// growFinderScan reallocates the finder-scan record and chain-outcome buffers
// for at least capacity records and rebinds the stages that reference them.
// The route admission budget covers only the initial capacity, so growth is
// opportunistic: a failed allocation leaves the old state intact and the
// caller keeps the bit-identical CPU row walk for the overflowed pass. The
// retained growth is reported through onRetainedAllocation so the pool charges it
// once the context returns to its free list.
func (b *gpuBinarizer) growFinderScan(capacity int) error {
	if capacity <= b.scanCapacity {
		return nil
	}
	records, err := b.device.NewBuffer(uint64(gpuFinderScanBufferSize(capacity)))
	if err != nil {
		return fmt.Errorf("jabcode: grow GPU finder scan records: %w", err)
	}
	outcomes, err := b.device.NewBuffer(uint64(gpuFinderChainBufferSize(capacity)))
	if err != nil {
		_ = records.Close()
		return fmt.Errorf("jabcode: grow GPU finder chain outcomes: %w", err)
	}
	scan, err := b.newStage(
		b.kernels.finderRowScan,
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, records),
		vulki.BindBuffer(2, b.scanParams),
	)
	if err != nil {
		_ = outcomes.Close()
		_ = records.Close()
		return fmt.Errorf("jabcode: rebind GPU finder row scan: %w", err)
	}
	// The swap is committed; displaced resources close best-effort because
	// the new state is already live and correct. The chain stages rebind
	// lazily in chainChannels against the new buffers.
	if b.scan.bindings != nil {
		_ = b.scan.bindings.Close()
	}
	if b.chain.bindings != nil {
		_ = b.chain.bindings.Close()
		b.chain = gpuBinarizerStage{}
	}
	if b.chainBSI.bindings != nil {
		_ = b.chainBSI.bindings.Close()
		b.chainBSI = gpuBinarizerStage{}
	}
	_ = b.scanRecords.Close()
	_ = b.chainOutcomes.Close()
	if b.onRetainedAllocation != nil {
		b.onRetainedAllocation(gpuFinderScanGrowthBytes(b.scanCapacity, capacity))
	}
	b.scan = scan
	b.scanRecords = records
	b.chainOutcomes = outcomes
	b.scanCapacity = capacity
	b.hostScanRecords = make([]byte, gpuFinderScanBufferSize(capacity))
	// Until a download overwrites it, the fresh host buffer must still parse
	// as overflowed so a failed retry keeps the CPU row walk.
	binary.LittleEndian.PutUint32(b.hostScanRecords, math.MaxUint32)
	b.hostChainOutcomes = make([]byte, gpuFinderChainBufferSize(capacity))
	return nil
}

// rescanFinderScan reruns the finder row scan and chain kernels over the
// still resident packed masks in a submission of their own. It is the second
// half of an overflow retry after growFinderScan; the caller downloads the
// refreshed records afterward.
func (b *gpuBinarizer) rescanFinderScan(
	width, height int,
	channelMask uint32,
	printLevels bool,
	rowStride int,
) (uint32, error) {
	recorder, err := b.device.NewRecorder()
	if err != nil {
		return 0, fmt.Errorf("jabcode: create GPU finder rescan recorder: %w", err)
	}
	defer recorder.Abort()
	chainChannels, err := b.recordFinderScan(recorder, width, height, channelMask, printLevels, rowStride)
	if err != nil {
		return 0, err
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return 0, fmt.Errorf("jabcode: run GPU finder rescan: %w", err)
	}
	return chainChannels, nil
}

// gpuFinderChainParams packs the finder chain kernel's parameters: the image
// shape, the print-slack flag and the palette classification bits, which stay
// authoritative on the host.
func gpuFinderChainParams(width, height, capacity int, printLevels bool) [gpuFinderChainParamsSize]byte {
	var params [gpuFinderChainParamsSize]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	binary.LittleEndian.PutUint32(params[8:], uint32(capacity))
	flags := uint32(0)
	if printLevels {
		flags |= 1
	}
	binary.LittleEndian.PutUint32(params[12:], flags)
	var classifyCurrent, classifyBSI, crossBits uint32
	for t := range 4 {
		coreIdx := fpCoreColorIndex(t)
		bsiIdx := bsiFamilyFinderCoreColors[t]
		for c := range 3 {
			if palette.Default[coreIdx*3+c] > 0 {
				classifyCurrent |= 1 << (t*3 + c)
			}
			if palette.Default[bsiIdx*3+c] > 0 {
				classifyBSI |= 1 << (t*3 + c)
			}
		}
	}
	if palette.Default[spec.FP3CoreColor*3] > 0 {
		crossBits |= 1
	}
	if palette.Default[spec.FP2CoreColor*3+2] > 0 {
		crossBits |= 2
	}
	binary.LittleEndian.PutUint32(params[16:], classifyCurrent)
	binary.LittleEndian.PutUint32(params[20:], classifyBSI)
	binary.LittleEndian.PutUint32(params[24:], crossBits)
	return params
}

func (b *gpuBinarizer) newStage(
	kernel func() (*vulki.Kernel, error),
	buffers ...vulki.BufferBinding,
) (gpuBinarizerStage, error) {
	shared, err := kernel()
	if err != nil {
		return gpuBinarizerStage{}, err
	}
	bindings, err := shared.NewBindings(buffers...)
	if err != nil {
		return gpuBinarizerStage{}, err
	}
	return gpuBinarizerStage{kernel: shared, bindings: bindings}, nil
}

func (b *gpuBinarizer) Binarize(bm *core.Bitmap, blkThs []float32, printLevels bool) ([3]*core.Bitmap, error) {
	var empty [3]*core.Bitmap
	if bm == nil || bm.Width <= 0 || bm.Height <= 0 || bm.Channels != 4 {
		return empty, fmt.Errorf("jabcode: GPU binarizer requires a non-empty packed RGBA bitmap")
	}
	if bm.Width > b.maxWidth || bm.Height > b.maxHeight {
		return empty, fmt.Errorf("jabcode: GPU binarizer image %dx%d exceeds configured maximum %dx%d", bm.Width, bm.Height, b.maxWidth, b.maxHeight)
	}
	pixelCount := bm.Width * bm.Height
	if len(bm.Pix) != pixelCount*4 {
		return empty, fmt.Errorf("jabcode: GPU binarizer requires a non-empty packed RGBA bitmap")
	}
	if blkThs != nil && len(blkThs) < 3 {
		return empty, fmt.Errorf("jabcode: GPU binarizer needs three fixed thresholds")
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || b.device == nil || b.device.Closed() {
		return empty, fmt.Errorf("jabcode: GPU binarizer is closed")
	}
	if b.input == nil || b.classify.bindings == nil {
		return empty, fmt.Errorf("jabcode: GPU binarizer has no host-input path")
	}

	params, thresholdData := gpuBinarizerInputs(bm, blkThs, printLevels)
	packedMasks := b.hostMasks[:((pixelCount+7)/8)*4]
	recorder, err := b.device.NewRecorder()
	if err != nil {
		return empty, fmt.Errorf("jabcode: create GPU binarizer recorder: %w", err)
	}
	defer recorder.Abort()
	phaseprobe.Count("upload.binarizer_input", len(bm.Pix))
	if err := recorder.Upload(b.input, 0, bm.Pix); err != nil {
		return empty, fmt.Errorf("jabcode: upload GPU binarizer image: %w", err)
	}
	if err := recorder.Upload(b.thresholds, 0, thresholdData); err != nil {
		return empty, fmt.Errorf("jabcode: upload GPU binarizer thresholds: %w", err)
	}
	if err := recorder.Update(b.params, 0, params); err != nil {
		return empty, fmt.Errorf("jabcode: update GPU binarizer parameters: %w", err)
	}
	if err := b.recordCompute(recorder, bm.Width, bm.Height); err != nil {
		return empty, err
	}
	phaseprobe.Count("download.packed_masks", len(packedMasks))
	if err := recorder.Download(b.packedMasks, 0, packedMasks); err != nil {
		return empty, fmt.Errorf("jabcode: download GPU binarizer masks: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, fmt.Errorf("jabcode: run GPU binarizer: %w", err)
	}
	return unpackGPUBinarizerMasks(bm, packedMasks), nil
}

// gpuMaskExpand maps twelve packed mask bits - four pixels carrying three
// channel bits each - to the three four-byte runs those pixels expand to.
// Every route that binarizes on the device expands a whole frame this way, so
// the byte-at-a-time form showed up as one of the largest consumers in the
// profile once routes stopped falling back to the CPU. One entry is twelve
// contiguous bytes, so a lookup touches a single cache line, and real masks run
// in long all-set and all-clear stretches, which keeps the hot entries
// resident.
var gpuMaskExpand = func() *[4096][3]uint32 {
	table := new([4096][3]uint32)
	for index := range table {
		for pixel := range 4 {
			for channel := range 3 {
				if index&(1<<(pixel*3+channel)) != 0 {
					table[index][channel] |= uint32(0xFF) << (pixel * 8)
				}
			}
		}
	}
	return table
}()

func unpackGPUBinarizerMasks(bm *core.Bitmap, packedMasks []byte) [3]*core.Bitmap {
	pixelCount := bm.Width * bm.Height
	var rgb [3]*core.Bitmap
	for channel := range rgb {
		rgb[channel] = newBinary(bm)
	}
	wordCount := (pixelCount + 7) / 8
	core.ParallelChunks(wordCount, 1024, func(lo, hi int) {
		red, green, blue := rgb[0].Pix, rgb[1].Pix, rgb[2].Pix
		pixel := lo * 8
		for word := lo; word < hi; word++ {
			packed := binary.LittleEndian.Uint32(packedMasks[word*4:])
			if pixel+8 <= pixelCount {
				low := &gpuMaskExpand[packed&0xFFF]
				high := &gpuMaskExpand[(packed>>12)&0xFFF]
				// One packed word is eight pixels, so each channel's whole
				// contribution is a single 64-bit store.
				binary.LittleEndian.PutUint64(red[pixel:], uint64(low[0])|uint64(high[0])<<32)
				binary.LittleEndian.PutUint64(green[pixel:], uint64(low[1])|uint64(high[1])<<32)
				binary.LittleEndian.PutUint64(blue[pixel:], uint64(low[2])|uint64(high[2])<<32)
				pixel += 8
				continue
			}
			for lane := 0; lane < 8 && pixel < pixelCount; lane++ {
				mask := packed & 7
				rgb[0].Pix[pixel] = b2byte(mask&1 != 0)
				rgb[1].Pix[pixel] = b2byte(mask&2 != 0)
				rgb[2].Pix[pixel] = b2byte(mask&4 != 0)
				packed >>= 3
				pixel++
			}
		}
	})
	return rgb
}

func (b *gpuBinarizer) recordCompute(recorder *vulki.Recorder, width, height int) error {
	return b.recordComputeWithClassifier(recorder, b.classify.bindings, width, height)
}

func (b *gpuBinarizer) recordComputeWithClassifier(
	recorder *vulki.Recorder,
	classifier *vulki.BindingSet,
	width, height int,
) error {
	pixelCount := width * height
	pixelGroups := vulki.Workgroups{
		X: uint32((width + gpuBinarizerWorkgroupWidth - 1) / gpuBinarizerWorkgroupWidth),
		Y: uint32((height + gpuBinarizerWorkgroupHeight - 1) / gpuBinarizerWorkgroupHeight),
		Z: 1,
	}
	if err := recorder.Dispatch(b.classify.kernel, classifier, pixelGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU RGB classifier: %w", err)
	}
	if err := recorder.Barrier(b.rawMasks); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU RGB classifier: %w", err)
	}
	if err := recorder.Dispatch(b.filter.kernel, b.filter.bindings, pixelGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU binary filter: %w", err)
	}
	if err := recorder.Barrier(b.finalMasks); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU binary filter: %w", err)
	}
	packGroups := vulki.Workgroups{
		X: uint32(((pixelCount+7)/8 + gpuPackWorkgroupSize - 1) / gpuPackWorkgroupSize),
		Y: 1,
		Z: 1,
	}
	if err := recorder.Dispatch(b.pack.kernel, b.pack.bindings, packGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU mask packer: %w", err)
	}
	return nil
}

func gpuBinarizerInputs(bm *core.Bitmap, blkThs []float32, printLevels bool) (params, thresholds []byte) {
	params = make([]byte, gpuBinarizerParamsSize)
	binary.LittleEndian.PutUint32(params[0:], uint32(bm.Width))
	binary.LittleEndian.PutUint32(params[4:], uint32(bm.Height))
	flags := uint32(0)
	if blkThs != nil {
		flags |= 1
		binary.LittleEndian.PutUint32(params[24:], math.Float32bits(blkThs[0]))
		binary.LittleEndian.PutUint32(params[28:], math.Float32bits(blkThs[1]))
		binary.LittleEndian.PutUint32(params[32:], math.Float32bits(blkThs[2]))
		binary.LittleEndian.PutUint32(params[8:], 1)
		binary.LittleEndian.PutUint32(params[12:], 1)
		binary.LittleEndian.PutUint32(params[16:], 1)
		thresholds = make([]byte, gpuThresholdCellSize)
	} else {
		bs := capInt(min(bm.Width, bm.Height)/binThresholdDivisor, binMinBlock, binMaxBlock)
		anchors, means, blocksX, blocksY := blockThresholds(bm, bs)
		binary.LittleEndian.PutUint32(params[8:], uint32(bs))
		binary.LittleEndian.PutUint32(params[12:], uint32(blocksX))
		binary.LittleEndian.PutUint32(params[16:], uint32(blocksY))
		thresholds = make([]byte, len(means)*gpuThresholdCellSize)
		for index := range means {
			for channel := range 3 {
				binary.LittleEndian.PutUint32(thresholds[index*gpuThresholdCellSize+channel*4:], math.Float32bits(float32(anchors[index][channel])))
				binary.LittleEndian.PutUint32(thresholds[index*gpuThresholdCellSize+16+channel*4:], math.Float32bits(float32(means[index][channel])))
			}
		}
	}
	if printLevels {
		flags |= 2
	}
	binary.LittleEndian.PutUint32(params[20:], flags)
	return params, thresholds
}

func (b *gpuBinarizer) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	return b.closeResources()
}

func (b *gpuBinarizer) closeResources() error {
	var closeErrors []error
	closeErrors = append(closeErrors, b.closeDirectional())
	// The stage kernels belong to the shared per-device set; only the binding
	// sets are this instance's to close.
	for _, stage := range []*gpuBinarizerStage{&b.chainBSI, &b.chain, &b.scan, &b.pack, &b.filter, &b.classify} {
		if stage.bindings != nil {
			closeErrors = append(closeErrors, stage.bindings.Close())
			stage.bindings = nil
		}
		stage.kernel = nil
	}
	for _, buffer := range []*vulki.Buffer{
		b.chainParams, b.chainOutcomes, b.scanParams, b.scanRecords,
		b.params, b.dirBatchOutcomes, b.dirBatchSummary, b.preservedMasks, b.packedMasks, b.finalMasks, b.rawMasks, b.thresholds, b.input,
		b.rowSummary, b.rowCompacted,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	b.chainParams = nil
	b.chainOutcomes = nil
	b.hostChainOutcomes = nil
	b.rowSummary = nil
	b.rowCompacted = nil
	b.hostRowSummary = nil
	b.hostRowCompacted = nil
	b.scanParams = nil
	b.scanRecords = nil
	b.hostScanRecords = nil
	b.params = nil
	b.packedMasks = nil
	b.finalMasks = nil
	b.rawMasks = nil
	b.thresholds = nil
	b.input = nil
	b.hostMasks = nil
	if b.ownsKernels {
		closeErrors = append(closeErrors, b.kernels.Close())
	}
	b.kernels = nil
	if b.ownsDevice && b.device != nil {
		closeErrors = append(closeErrors, b.device.Close())
	}
	b.device = nil
	return errors.Join(closeErrors...)
}
