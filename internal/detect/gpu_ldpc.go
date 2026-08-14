//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/phaseprobe"
)

//go:embed shaders/ldpc_hard.wgsl
var ldpcHardWGSL string

//go:embed shaders/ldpc_soft.wgsl
var ldpcSoftWGSL string

//go:embed shaders/ldpc_soft_graph.wgsl
var ldpcSoftGraphWGSL string

//go:embed shaders/ldpc_soft_prepare.wgsl
var ldpcSoftPrepareWGSL string

//go:embed shaders/ldpc_matrix.wgsl
var ldpcMatrixWGSL string

// Parameter word indices, matching ldpc_hard.wgsl.
const (
	gpuLDPCParamLength        = 0
	gpuLDPCParamHeight        = 1
	gpuLDPCParamRank          = 2
	gpuLDPCParamNet           = 3
	gpuLDPCParamBlocks        = 4
	gpuLDPCParamRowDegree     = 5
	gpuLDPCParamTailBlock     = 6
	gpuLDPCParamTailLength    = 7
	gpuLDPCParamTailHeight    = 8
	gpuLDPCParamTailRank      = 9
	gpuLDPCParamTailNet       = 10
	gpuLDPCParamTailRowDegree = 11
	gpuLDPCParamTailRowBase   = 12
	gpuLDPCParamAdmission     = 13
	gpuLDPCParamRowBase       = 14
	gpuLDPCParamWords         = 16
)

// gpuLDPCRowWords is where the trailing block's parity rows start in the shared
// rows buffer. Each set needs at most one row per bit of its block, and a row
// holds at most the code's row weight; sixteen slots per bit covers every legal
// weight with room for the padding sentinel.
const gpuLDPCRowWords = gpuLDPCMaxSub * 16

const (
	gpuLDPCMatrixMaxStride   = (gpuLDPCMaxSub + 31) / 32
	gpuLDPCMatrixDenseWords  = gpuLDPCMaxSub * gpuLDPCMatrixMaxStride
	gpuLDPCMatrixSourceWords = gpuLDPCMaxSub * 16
	gpuLDPCMatrixStateBase   = gpuLDPCMatrixDenseWords + gpuLDPCMatrixSourceWords +
		gpuLDPCMaxSub + gpuLDPCMatrixMaxStride + gpuLDPCMaxSub +
		2*gpuLDPCMaxSub + gpuLDPCMaxSub + gpuLDPCMaxSub
	gpuLDPCMatrixScratchWords = gpuLDPCMatrixStateBase + gpuLDPCMatrixStateWords
	gpuLDPCMatrixCacheWords   = 2 * 8
)

// gpuLDPCMatrixSource is the matrix builder compiled for one slot and phase. The
// two selectors are prepended rather than uploaded because they decide which
// control words a phase reads and which branches compile away entirely.
func gpuLDPCMatrixSource(slot, phase uint32) string {
	return fmt.Sprintf(
		"const MATRIX_SLOT: u32 = %du;\nconst MATRIX_PHASE: u32 = %du;\n", slot, phase,
	) + ldpcMatrixWGSL
}

// The phases of the blocked matrix build, matching MATRIX_PHASE in
// ldpc_matrix.wgsl.
const (
	gpuLDPCMatrixPhaseSetup = iota
	gpuLDPCMatrixPhasePanel
	gpuLDPCMatrixPhaseApply
	gpuLDPCMatrixPhaseFinish
)

// The blocked elimination's shape, matching ldpc_matrix.wgsl. The panel width is
// bounded by the 32 bits of the per-row selection mask the apply phase carries
// and by the workgroup memory the staged panel occupies. Panel steps are
// recorded to the worst case because the selected matrix height is device-side
// control the host has not downloaded; the steps past the real height cost one
// workgroup that reads a word and exits, plus an apply dispatch the panel has
// already zeroed.
const (
	gpuLDPCMatrixPanel      = 32
	gpuLDPCMatrixPanelSteps = (gpuLDPCMaxSub + gpuLDPCMatrixPanel - 1) / gpuLDPCMatrixPanel

	gpuLDPCMatrixStateApplyDims = 9
	gpuLDPCMatrixStateWords     = 12 + 2*gpuLDPCMatrixPanel

	// The apply phase reads its grid out of the workspace the panel phase
	// writes, so the indirect source is the scratch buffer at this byte offset.
	gpuLDPCMatrixApplyIndirectOffset = (gpuLDPCMatrixStateBase + gpuLDPCMatrixStateApplyDims) * 4
)

// gpuLDPCMaxSub must match MAX_SUB in ldpc_hard.wgsl: the bound on one gross
// sub-block. The sub-block split runs until a block is under 2700 bits, so no
// block reaches this.
const gpuLDPCMaxSub = 2816

// gpuLDPCMaxBlocks bounds how many sub-blocks one correction may hold. A
// symbol's gross payload divided into blocks under 2700 bits does not approach
// this at any legal version.
const gpuLDPCMaxBlocks = 64

// The correction buffer keeps compact message bits at its front and a fixed
// evidence tail beyond every legal message location. Hard and soft workgroups
// reduce into these atomics without another result buffer or host crossing.
const (
	gpuLDPCEvidenceInitial = iota
	gpuLDPCEvidenceHardResidual
	gpuLDPCEvidenceCorrections
	gpuLDPCEvidenceSoftUsed
	gpuLDPCEvidenceSoftResidual
	gpuLDPCEvidenceSoftIterations
	gpuLDPCEvidenceWords

	gpuLDPCEvidenceBase = gpuLDPCMaxBlocks + gpuLDPCMaxBlocks*gpuLDPCMaxSub
)

// gpuLDPCRetainedBytes is what the corrector holds on the device: the parity
// rows, the staged codeword, its output, the parameter block, and the sparse
// soft-retry state that is used only after a hard syndrome failure.
const gpuLDPCRetainedBytes = 2*gpuLDPCRowWords*4 +
	gpuLDPCMaxBlocks*gpuLDPCMaxSub*4 +
	(gpuLDPCEvidenceBase+gpuLDPCEvidenceWords)*4 +
	gpuLDPCParamWords*4 + gpuLDPCMatrixScratchWords*4 +
	gpuLDPCMatrixCacheWords*4 + gpuLDPCSoftRetainedBytes

// gpuLDPCPlan is one correction request: the parity rows of the code, and the
// shape of the sub-blocks the codeword splits into.
//
// rows is the flattened parity-check matrix, rowDegree columns per row. It
// depends only on the code parameters and never on the image, so a caller that
// decodes many symbols under one code uploads it once.
type gpuLDPCPlan struct {
	rows      []uint32
	rowDegree int
	length    int
	height    int
	rank      int
	net       int
	blocks    int

	// tail describes the trailing block when the codeword does not divide into
	// equal sub-blocks, which is the ordinary case: the uniform length is
	// rounded down to a multiple of the row weight and the last block absorbs
	// the remainder under a parity-check matrix built for its own length. A
	// zero tailLength means the split came out even.
	tailRows      []uint32
	tailRowDegree int
	tailLength    int
	tailHeight    int
	tailRank      int
	tailNet       int
}

func (plan gpuLDPCPlan) valid() bool {
	if plan.rowDegree <= 0 || plan.length <= 0 || plan.length > gpuLDPCMaxSub ||
		plan.height <= 0 || plan.rank < 0 || plan.rank > plan.length ||
		plan.net <= 0 || plan.net > plan.length ||
		plan.blocks <= 0 || plan.blocks > gpuLDPCMaxBlocks ||
		len(plan.rows) > gpuLDPCRowWords ||
		len(plan.rows) < plan.height*plan.rowDegree {
		return false
	}
	if plan.tailLength == 0 {
		return true
	}
	return plan.tailRowDegree > 0 && plan.tailLength <= gpuLDPCMaxSub &&
		plan.tailHeight > 0 && plan.tailRank >= 0 && plan.tailRank <= plan.tailLength &&
		plan.tailNet > 0 && plan.tailNet <= plan.tailLength &&
		len(plan.tailRows) <= gpuLDPCRowWords &&
		len(plan.tailRows) >= plan.tailHeight*plan.tailRowDegree
}

// netWords is how many message-bit words the correction writes. Every block
// starts at its own multiple of the uniform net length, and a longer trailing
// block simply runs past that stride.
func (plan gpuLDPCPlan) netWords() int {
	last := plan.net
	if plan.tailLength != 0 {
		last = plan.tailNet
	}
	return (plan.blocks-1)*plan.net + last
}

// initializeLDPC allocates the correction buffers and compiles the kernel with
// the rest of the resident stage set, so the compile lands in warm-up rather
// than on the decode call.
func (resident *gpuResidentBinarizer) initializeLDPC() error {
	var err error
	resident.ldpcParams, err = resident.device.NewBuffer(gpuLDPCParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC parameters: %w", err)
	}
	resident.ldpcBits, err = resident.device.NewBuffer(gpuLDPCMaxBlocks * gpuLDPCMaxSub * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC codeword: %w", err)
	}
	resident.ldpcNet, err = resident.device.NewBuffer(
		(gpuLDPCEvidenceBase + gpuLDPCEvidenceWords) * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC output: %w", err)
	}
	resident.ldpcRows, err = resident.device.NewBuffer(2 * gpuLDPCRowWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC parity rows: %w", err)
	}
	resident.ldpcReliability, err = resident.device.NewBuffer(gpuLDPCMaxBlocks * gpuLDPCMaxSub * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC reliability: %w", err)
	}
	resident.ldpcSoftGraph, err = resident.device.NewBuffer(gpuLDPCSoftColumnBufferWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU soft LDPC graph: %w", err)
	}
	resident.ldpcMessages, err = resident.device.NewBuffer(gpuLDPCSoftMessageWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU soft LDPC messages: %w", err)
	}
	resident.ldpcSoftIndirect, err = resident.device.NewBuffer(gpuLDPCSoftIndirectWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU soft LDPC indirect dispatches: %w", err)
	}
	resident.ldpcKernel, err = resident.kernels.ldpcHard()
	if err != nil {
		return err
	}
	resident.ldpcSoftKernel, err = resident.kernels.ldpcSoft()
	if err != nil {
		return err
	}
	resident.ldpcSoftGraphKernel, err = resident.kernels.ldpcSoftGraph()
	if err != nil {
		return err
	}
	resident.ldpcSoftPrepareKernel, err = resident.kernels.ldpcSoftPrepare()
	if err != nil {
		return err
	}
	resident.ldpcBindings, err = resident.ldpcKernel.NewBindings(
		vulki.BindBuffer(0, resident.ldpcRows),
		vulki.BindBuffer(1, resident.ldpcBits),
		vulki.BindBuffer(2, resident.ldpcParams),
		vulki.BindBuffer(3, resident.ldpcNet),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU LDPC correction: %w", err)
	}
	resident.ldpcSoftBindings, err = resident.ldpcSoftKernel.NewBindings(
		vulki.BindBuffer(0, resident.ldpcRows),
		vulki.BindBuffer(1, resident.ldpcSoftGraph),
		vulki.BindBuffer(2, resident.ldpcBits),
		vulki.BindBuffer(3, resident.ldpcReliability),
		vulki.BindBuffer(4, resident.ldpcParams),
		vulki.BindBuffer(5, resident.ldpcMessages),
		vulki.BindBuffer(6, resident.ldpcNet),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU soft LDPC correction: %w", err)
	}
	resident.ldpcSoftGraphBindings, err = resident.ldpcSoftGraphKernel.NewBindings(
		vulki.BindBuffer(0, resident.ldpcRows),
		vulki.BindBuffer(1, resident.ldpcParams),
		vulki.BindBuffer(2, resident.ldpcNet),
		vulki.BindBuffer(3, resident.ldpcSoftGraph),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU soft LDPC graph: %w", err)
	}
	return nil
}

// initializeLDPCMatrix gives the payload route one packed elimination
// workspace shared by its regular and optional trailing matrices. The two
// fixed pipelines select those slots without a per-frame selector upload.
func (resident *gpuResidentBinarizer) initializeLDPCMatrix() error {
	var err error
	resident.ldpcMatrixScratch, err = resident.device.NewBuffer(gpuLDPCMatrixScratchWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC matrix workspace: %w", err)
	}
	resident.ldpcMatrixCache, err = resident.device.NewBuffer(gpuLDPCMatrixCacheWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC matrix cache: %w", err)
	}
	stage := func(
		slot, phase uint32,
		kernel **vulki.Kernel,
		set **vulki.BindingSet,
		what string,
	) error {
		built, err := resident.kernels.ldpcMatrix(slot, phase)
		if err != nil {
			return err
		}
		bound, err := built.NewBindings(
			vulki.BindBuffer(0, resident.payloadParams),
			vulki.BindBuffer(1, resident.ldpcParams),
			vulki.BindBuffer(2, resident.ldpcRows),
			vulki.BindBuffer(3, resident.ldpcMatrixScratch),
			vulki.BindBuffer(4, resident.ldpcMatrixCache),
		)
		if err != nil {
			return fmt.Errorf("jabcode: bind resident GPU %s: %w", what, err)
		}
		*kernel = built
		*set = bound
		return nil
	}
	for _, built := range []struct {
		slot, phase uint32
		kernel      **vulki.Kernel
		set         **vulki.BindingSet
		what        string
	}{
		{0, gpuLDPCMatrixPhaseSetup, &resident.ldpcMatrixSetupKernel,
			&resident.ldpcMatrixSetupBindings, "LDPC matrix source"},
		{1, gpuLDPCMatrixPhaseSetup, &resident.ldpcTailMatrixSetupKernel,
			&resident.ldpcTailMatrixSetupBindings, "trailing LDPC matrix source"},
		{0, gpuLDPCMatrixPhasePanel, &resident.ldpcMatrixPanelKernel,
			&resident.ldpcMatrixPanelBindings, "LDPC matrix panel"},
		{0, gpuLDPCMatrixPhaseApply, &resident.ldpcMatrixApplyKernel,
			&resident.ldpcMatrixApplyBindings, "LDPC matrix panel apply"},
		{0, gpuLDPCMatrixPhaseFinish, &resident.ldpcMatrixFinishKernel,
			&resident.ldpcMatrixFinishBindings, "LDPC matrix arrangement"},
		{1, gpuLDPCMatrixPhaseFinish, &resident.ldpcTailMatrixFinishKernel,
			&resident.ldpcTailMatrixFinishBindings, "trailing LDPC matrix arrangement"},
	} {
		if err := stage(built.slot, built.phase, built.kernel, built.set, built.what); err != nil {
			return err
		}
	}
	resident.ldpcMatrixCacheDirty = true
	return nil
}

// recordGPULDPCMatrix records one complete blocked matrix build for a slot.
//
// The panel step is dispatched at its worst-case count rather than the selected
// height, because that height is control the device derives and the host has not
// downloaded. A step past the end reads one state word and exits, and clears the
// apply grid so the recorded apply steps behind it become zero work.
func recordGPULDPCMatrix(
	recorder *vulki.Recorder,
	resident *gpuResidentBinarizer,
	setup, finish *vulki.Kernel,
	setupBindings, finishBindings *vulki.BindingSet,
	active *vulki.Buffer,
	what string,
) error {
	workspace := []*vulki.Buffer{
		resident.ldpcRows, resident.ldpcParams,
		resident.ldpcMatrixScratch, resident.ldpcMatrixCache,
	}
	if err := recordGPUOneWorkgroup(recorder, setup, setupBindings, active); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU %s source: %w", what, err)
	}
	if err := recorder.Barrier(workspace...); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU %s source: %w", what, err)
	}
	for range gpuLDPCMatrixPanelSteps {
		if err := recordGPUOneWorkgroup(
			recorder, resident.ldpcMatrixPanelKernel, resident.ldpcMatrixPanelBindings, active,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch GPU %s panel: %w", what, err)
		}
		if err := recorder.Barrier(workspace...); err != nil {
			return fmt.Errorf("jabcode: synchronize GPU %s panel: %w", what, err)
		}
		if err := recorder.DispatchIndirect(
			resident.ldpcMatrixApplyKernel, resident.ldpcMatrixApplyBindings,
			resident.ldpcMatrixScratch, gpuLDPCMatrixApplyIndirectOffset,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch GPU %s panel apply: %w", what, err)
		}
		if err := recorder.Barrier(workspace...); err != nil {
			return fmt.Errorf("jabcode: synchronize GPU %s panel apply: %w", what, err)
		}
	}
	if err := recordGPUOneWorkgroup(recorder, finish, finishBindings, active); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU %s arrangement: %w", what, err)
	}
	if err := recorder.Barrier(workspace...); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU %s arrangement: %w", what, err)
	}
	return nil
}

// CorrectLDPCHard runs hard-decision LDPC correction for every sub-block of a
// codeword at once, and hands back the compacted message bits with the
// per-block syndrome verdict.
//
// The codeword arrives as one bit per byte, which is the representation the
// host encoder and decoder already use throughout. ok is false when any block's
// post-correction syndrome is still unsatisfied, which is the reference's own
// give-up condition and the only signal that the stream is garbage of the right
// length: hard LDPC has no payload integrity check underneath it.
func (resident *gpuResidentBinarizer) CorrectLDPCHard(
	plan gpuLDPCPlan,
	codeword []byte,
) (dec []byte, ok bool, err error) {
	if resident == nil || resident.closed || resident.ldpcBindings == nil {
		return nil, false, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if !plan.valid() {
		return nil, false, fmt.Errorf("jabcode: GPU LDPC plan is out of range")
	}
	if len(codeword) < plan.blocks*plan.length {
		return nil, false, fmt.Errorf("jabcode: GPU LDPC codeword is shorter than its blocks")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	// This diagnostic entry point uploads caller-supplied rows into the payload
	// row buffer. Its next resident payload use must rebuild rather than trust a
	// cache record that names the rows this call replaces.
	resident.ldpcMatrixCacheDirty = true

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, false, fmt.Errorf("jabcode: create GPU LDPC recorder: %w", err)
	}
	defer recorder.Abort()

	if err := gpuLDPCUploadRows(recorder, resident.ldpcRows, plan); err != nil {
		return nil, false, err
	}
	bits := make([]byte, plan.blocks*plan.length*4)
	for at := range plan.blocks * plan.length {
		if codeword[at] != 0 {
			binary.LittleEndian.PutUint32(bits[at*4:], 1)
		}
	}
	if err := recordGPUUpdate(recorder, "upload.ldpc_bits", resident.ldpcBits, 0, bits); err != nil {
		return nil, false, fmt.Errorf("jabcode: upload GPU LDPC codeword: %w", err)
	}
	params := gpuLDPCParams(plan)
	if err := recordGPUUpdate(
		recorder, "upload.ldpc_params", resident.ldpcParams, 0, params[:],
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: update GPU LDPC parameters: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcRows, resident.ldpcBits, resident.ldpcParams); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU LDPC inputs: %w", err)
	}
	if err := resident.clearLDPCEvidence(recorder); err != nil {
		return nil, false, err
	}
	if err := recorder.Dispatch(
		resident.ldpcKernel,
		resident.ldpcBindings,
		vulki.Workgroups{X: uint32(plan.blocks), Y: 1, Z: 1},
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: dispatch GPU LDPC correction: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU LDPC output: %w", err)
	}
	out := make([]byte, (plan.blocks+plan.netWords())*4)
	phaseprobe.Count("download.ldpc_net", len(out))
	if err := recorder.Download(resident.ldpcNet, 0, out); err != nil {
		return nil, false, fmt.Errorf("jabcode: record GPU LDPC download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, false, fmt.Errorf("jabcode: run GPU LDPC correction: %w", err)
	}
	return gpuLDPCResult(plan, out, plan.netWords())
}

func (resident *gpuResidentBinarizer) clearLDPCEvidence(recorder *vulki.Recorder) error {
	if err := recorder.Fill(
		resident.ldpcNet,
		uint64(gpuLDPCEvidenceBase*4),
		uint64(gpuLDPCEvidenceWords*4),
		0,
	); err != nil {
		return fmt.Errorf("jabcode: clear GPU LDPC evidence: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU LDPC evidence: %w", err)
	}
	return nil
}

// gpuLDPCResult reads the correction's status words and message bits back out
// of one download.
func gpuLDPCResult(plan gpuLDPCPlan, out []byte, bits int) (dec []byte, ok bool, err error) {
	ok = true
	for block := range plan.blocks {
		if binary.LittleEndian.Uint32(out[block*4:]) != 0 {
			ok = false
		}
	}
	dec = make([]byte, bits)
	for at := range dec {
		dec[at] = byte(binary.LittleEndian.Uint32(out[(plan.blocks+at)*4:]) & 1)
	}
	return dec, ok, nil
}

// gpuLDPCUploadRows stages both parity-check matrices a correction may need.
// They depend only on the code parameters, never on the image.
func gpuLDPCUploadRows(recorder *vulki.Recorder, buffer *vulki.Buffer, plan gpuLDPCPlan) error {
	rows := make([]byte, (len(plan.rows)+len(plan.tailRows))*4)
	for at, column := range plan.rows {
		binary.LittleEndian.PutUint32(rows[at*4:], column)
	}
	tailAt := len(plan.rows) * 4
	for at, column := range plan.tailRows {
		binary.LittleEndian.PutUint32(rows[tailAt+at*4:], column)
	}
	if err := recordGPUUpdate(
		recorder, "upload.ldpc_matrix", buffer, 0, rows[:len(plan.rows)*4],
	); err != nil {
		return fmt.Errorf("jabcode: upload GPU LDPC parity rows: %w", err)
	}
	if len(plan.tailRows) == 0 {
		return nil
	}
	if err := recordGPUUpdate(
		recorder, "upload.ldpc_matrix", buffer, gpuLDPCRowWords*4, rows[tailAt:],
	); err != nil {
		return fmt.Errorf("jabcode: upload GPU LDPC trailing parity rows: %w", err)
	}
	return nil
}

func gpuLDPCParams(plan gpuLDPCPlan) [gpuLDPCParamWords * 4]byte {
	var params [gpuLDPCParamWords * 4]byte
	put := func(index, value int) {
		binary.LittleEndian.PutUint32(params[index*4:], uint32(value))
	}
	put(gpuLDPCParamLength, plan.length)
	put(gpuLDPCParamHeight, plan.height)
	put(gpuLDPCParamRank, plan.rank)
	put(gpuLDPCParamNet, plan.net)
	put(gpuLDPCParamBlocks, plan.blocks)
	put(gpuLDPCParamRowDegree, plan.rowDegree)
	// An even split has no trailing block, which the kernel reads as a block
	// index no workgroup can hold.
	put(gpuLDPCParamTailBlock, plan.blocks)
	if plan.tailLength != 0 {
		put(gpuLDPCParamTailBlock, plan.blocks-1)
		put(gpuLDPCParamTailLength, plan.tailLength)
		put(gpuLDPCParamTailHeight, plan.tailHeight)
		put(gpuLDPCParamTailRank, plan.tailRank)
		put(gpuLDPCParamTailNet, plan.tailNet)
		put(gpuLDPCParamTailRowDegree, plan.tailRowDegree)
		put(gpuLDPCParamTailRowBase, gpuLDPCRowWords)
	}
	return params
}
