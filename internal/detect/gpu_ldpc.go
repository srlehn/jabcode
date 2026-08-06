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

// Parameter word indices, matching ldpc_hard.wgsl.
const (
	gpuLDPCParamLength    = 0
	gpuLDPCParamHeight    = 1
	gpuLDPCParamRank      = 2
	gpuLDPCParamNet       = 3
	gpuLDPCParamBlocks    = 4
	gpuLDPCParamRowDegree = 5
	gpuLDPCParamWords     = 8
)

// gpuLDPCMaxSub must match MAX_SUB in ldpc_hard.wgsl: the bound on one gross
// sub-block. The sub-block split runs until a block is under 2700 bits, so no
// block reaches this.
const gpuLDPCMaxSub = 2816

// gpuLDPCMaxBlocks bounds how many sub-blocks one correction may hold. A
// symbol's gross payload divided into blocks under 2700 bits does not approach
// this at any legal version.
const gpuLDPCMaxBlocks = 64

// gpuLDPCRetainedBytes is what the corrector holds on the device: the parity
// rows, the staged codeword, its output, and the parameter block.
const gpuLDPCRetainedBytes = gpuLDPCMaxSub*16*4 +
	gpuLDPCMaxBlocks*gpuLDPCMaxSub*4 +
	(gpuLDPCMaxBlocks+gpuLDPCMaxBlocks*gpuLDPCMaxSub)*4 +
	gpuLDPCParamWords*4

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
}

func (plan gpuLDPCPlan) valid() bool {
	return plan.rowDegree > 0 && plan.length > 0 && plan.length <= gpuLDPCMaxSub &&
		plan.height > 0 && plan.rank >= 0 && plan.rank <= plan.length &&
		plan.net > 0 && plan.net <= plan.length &&
		plan.blocks > 0 && plan.blocks <= gpuLDPCMaxBlocks &&
		len(plan.rows) >= plan.height*plan.rowDegree
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
		(gpuLDPCMaxBlocks + gpuLDPCMaxBlocks*gpuLDPCMaxSub) * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC output: %w", err)
	}
	resident.ldpcRows, err = resident.device.NewBuffer(gpuLDPCMaxSub * 16 * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU LDPC parity rows: %w", err)
	}
	resident.ldpcKernel, err = resident.kernels.ldpcHard()
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

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, false, fmt.Errorf("jabcode: create GPU LDPC recorder: %w", err)
	}
	defer recorder.Abort()

	rows := make([]byte, len(plan.rows)*4)
	for at, column := range plan.rows {
		binary.LittleEndian.PutUint32(rows[at*4:], column)
	}
	if err := recorder.Update(resident.ldpcRows, 0, rows); err != nil {
		return nil, false, fmt.Errorf("jabcode: upload GPU LDPC parity rows: %w", err)
	}
	bits := make([]byte, plan.blocks*plan.length*4)
	for at := range plan.blocks * plan.length {
		if codeword[at] != 0 {
			binary.LittleEndian.PutUint32(bits[at*4:], 1)
		}
	}
	if err := recorder.Update(resident.ldpcBits, 0, bits); err != nil {
		return nil, false, fmt.Errorf("jabcode: upload GPU LDPC codeword: %w", err)
	}
	params := gpuLDPCParams(plan)
	if err := recorder.Update(resident.ldpcParams, 0, params[:]); err != nil {
		return nil, false, fmt.Errorf("jabcode: update GPU LDPC parameters: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcRows, resident.ldpcBits, resident.ldpcParams); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU LDPC inputs: %w", err)
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
	out := make([]byte, (plan.blocks+plan.blocks*plan.net)*4)
	phaseprobe.Count("download.ldpc_net", len(out))
	if err := recorder.Download(resident.ldpcNet, 0, out); err != nil {
		return nil, false, fmt.Errorf("jabcode: record GPU LDPC download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, false, fmt.Errorf("jabcode: run GPU LDPC correction: %w", err)
	}

	ok = true
	for block := range plan.blocks {
		if binary.LittleEndian.Uint32(out[block*4:]) != 0 {
			ok = false
		}
	}
	dec = make([]byte, plan.blocks*plan.net)
	for at := range dec {
		dec[at] = byte(binary.LittleEndian.Uint32(out[(plan.blocks+at)*4:]) & 1)
	}
	return dec, ok, nil
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
	return params
}
