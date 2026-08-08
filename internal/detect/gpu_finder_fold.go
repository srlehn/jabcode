//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

//go:embed shaders/finder_fold.wgsl
var finderFoldWGSL string

// Record and record layout, matching finder_fold.wgsl.
const (
	gpuFinderFoldCandidateWords = 6
	gpuFinderFoldPatternWords   = 6
	gpuFinderFoldRecordWords    = 8
	gpuFinderFoldParamWords     = 4

	gpuFinderFoldRecordTotal     = 0
	gpuFinderFoldRecordTypeCount = 1
	gpuFinderFoldRecordDropped   = 5
)

// gpuFinderFoldRetainedBytes is what the fold holds on the device for the life
// of a route context.
const gpuFinderFoldRetainedBytes = (gpuFinderFoldParamWords +
	gpuFinderDirectionalCompactCapacity*gpuFinderFoldCandidateWords +
	maxFinderPatterns*gpuFinderFoldPatternWords + gpuFinderFoldRecordWords) * 4

// initializeFinderFold allocates the fold's buffers and compiles its kernel
// with the rest of the resident stage set.
func (resident *gpuResidentBinarizer) initializeFinderFold() error {
	var err error
	resident.foldParams, err = resident.device.NewBuffer(gpuFinderFoldParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold parameters: %w", err)
	}
	resident.foldCandidates, err = resident.device.NewBuffer(
		gpuFinderDirectionalCompactCapacity * gpuFinderFoldCandidateWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold candidates: %w", err)
	}
	resident.foldPatterns, err = resident.device.NewBuffer(
		maxFinderPatterns * gpuFinderFoldPatternWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold patterns: %w", err)
	}
	resident.foldRecord, err = resident.device.NewBuffer(gpuFinderFoldRecordWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold record: %w", err)
	}
	resident.foldKernel, err = resident.kernels.finderFold()
	if err != nil {
		return err
	}
	resident.foldBindings, err = resident.foldKernel.NewBindings(
		vulki.BindBuffer(0, resident.foldParams),
		vulki.BindBuffer(1, resident.foldCandidates),
		vulki.BindBuffer(2, resident.foldPatterns),
		vulki.BindBuffer(3, resident.foldRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder fold: %w", err)
	}
	return nil
}

// gpuFinderFoldCandidate is one admitted candidate in the terms the fold reads.
// The chain's flags are not among them: a candidate reaching the fold has
// already passed them, so carrying them further would only invite the fold to
// second-guess a decision made where the evidence was.
type gpuFinderFoldCandidate struct {
	Typ        int
	Direction  int
	Centre     core.PointF
	ModuleSize float64
}

// gpuFinderFoldResult is the merged pattern list and the counts the caller
// tracks alongside it.
type gpuFinderFoldResult struct {
	Patterns  []FinderPattern
	TypeCount [4]int
	// Dropped counts candidates that found no slot because the list was full.
	// The host fold has no bound of its own, so any drop means the caller fed
	// it more distinct patterns than its own stop rule allows.
	Dropped int
}

// FoldFinderCandidates merges directional candidates into the accumulated
// pattern list where they already lie.
//
// The order of candidates is the caller's and is reproduced exactly: the fold's
// result depends on it, because a merge moves the entry it merged into and so
// changes what the next candidate matches.
//
// The arithmetic is f32 here and f64 on the host, so merged centres can differ
// in the last place and a candidate sitting exactly on the merge radius can
// fall the other way. That is the accepted trade for this route - decode
// outcomes are the gate, not bit parity - and it is why the comparison against
// the host is a tolerance and the census is what actually decides.
func (resident *gpuResidentBinarizer) FoldFinderCandidates(
	candidates []gpuFinderFoldCandidate,
) (gpuFinderFoldResult, error) {
	var result gpuFinderFoldResult
	if resident == nil || resident.closed || resident.foldBindings == nil {
		return result, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if len(candidates) > gpuFinderDirectionalCompactCapacity {
		return result, fmt.Errorf("jabcode: GPU finder fold takes up to %d candidates, got %d",
			gpuFinderDirectionalCompactCapacity, len(candidates))
	}
	if len(candidates) == 0 {
		return result, nil
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return result, fmt.Errorf("jabcode: create GPU finder fold recorder: %w", err)
	}
	defer recorder.Abort()

	var params [gpuFinderFoldParamWords * 4]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(len(candidates)))
	if err := recorder.Update(resident.foldParams, 0, params[:]); err != nil {
		return result, fmt.Errorf("jabcode: update GPU finder fold parameters: %w", err)
	}
	packed := make([]byte, len(candidates)*gpuFinderFoldCandidateWords*4)
	for i, candidate := range candidates {
		at := i * gpuFinderFoldCandidateWords * 4
		put := func(word int, value uint32) {
			binary.LittleEndian.PutUint32(packed[at+word*4:], value)
		}
		put(1, uint32(candidate.Typ))
		put(2, uint32(int32(candidate.Direction)))
		put(3, math.Float32bits(float32(candidate.Centre.X)))
		put(4, math.Float32bits(float32(candidate.Centre.Y)))
		put(5, math.Float32bits(float32(candidate.ModuleSize)))
	}
	// A recorded buffer update is bounded at 64 KB by the command itself, and a
	// full candidate list is several times that, so it goes in chunks. The
	// chunk size stays word aligned because the whole record layout is.
	const updateChunk = 64 << 10
	for at := 0; at < len(packed); at += updateChunk {
		end := min(at+updateChunk, len(packed))
		if err := recorder.Update(resident.foldCandidates, uint64(at), packed[at:end]); err != nil {
			return result, fmt.Errorf("jabcode: update GPU finder fold candidates: %w", err)
		}
	}
	if err := recorder.Barrier(resident.foldParams, resident.foldCandidates); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder fold inputs: %w", err)
	}
	if err := recorder.Dispatch(
		resident.foldKernel, resident.foldBindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU finder fold: %w", err)
	}
	if err := recorder.Barrier(resident.foldPatterns, resident.foldRecord); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder fold output: %w", err)
	}

	record := make([]byte, gpuFinderFoldRecordWords*4)
	patterns := make([]byte, maxFinderPatterns*gpuFinderFoldPatternWords*4)
	if err := recorder.Download(resident.foldRecord, 0, record); err != nil {
		return result, fmt.Errorf("jabcode: record GPU finder fold record download: %w", err)
	}
	if err := recorder.Download(resident.foldPatterns, 0, patterns); err != nil {
		return result, fmt.Errorf("jabcode: record GPU finder fold pattern download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return result, fmt.Errorf("jabcode: run GPU finder fold: %w", err)
	}
	return parseGPUFinderFold(record, patterns)
}

func parseGPUFinderFold(record, patterns []byte) (gpuFinderFoldResult, error) {
	var result gpuFinderFoldResult
	word := func(buf []byte, index int) uint32 {
		return binary.LittleEndian.Uint32(buf[index*4:])
	}
	total := int(word(record, gpuFinderFoldRecordTotal))
	if total > maxFinderPatterns {
		return result, fmt.Errorf("jabcode: GPU finder fold reported %d patterns", total)
	}
	result.Dropped = int(word(record, gpuFinderFoldRecordDropped))
	for typ := range result.TypeCount {
		result.TypeCount[typ] = int(word(record, gpuFinderFoldRecordTypeCount+typ))
	}
	result.Patterns = make([]FinderPattern, total)
	for i := range result.Patterns {
		at := i * gpuFinderFoldPatternWords
		result.Patterns[i] = FinderPattern{
			Typ:        int(word(patterns, at)),
			direction:  int(int32(word(patterns, at+1))),
			Center:     core.PointF{X: foldFloat(patterns, at+2), Y: foldFloat(patterns, at+3)},
			ModuleSize: foldFloat(patterns, at+4),
			FoundCount: int(word(patterns, at+5)),
		}
	}
	return result, nil
}

func foldFloat(buf []byte, index int) float64 {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[index*4:])))
}
