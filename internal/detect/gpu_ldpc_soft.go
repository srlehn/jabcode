//go:build !js

package detect

import (
	"fmt"

	"github.com/srlehn/jabcode/internal/ecc"
)

// A payload code has wc < wr <= 11, so no variable participates in more than
// ten checks. Keeping this format bound beside the allocation makes a future
// parity construction fail closed instead of indexing beyond resident scratch.
const gpuLDPCSoftMaxColumnDegree = 10

const (
	gpuLDPCSoftMaxEdges          = gpuLDPCMaxSub * gpuLDPCSoftMaxColumnDegree
	gpuLDPCSoftColumnWords       = gpuLDPCMaxSub * (1 + gpuLDPCSoftMaxColumnDegree)
	gpuLDPCSoftColumnBufferWords = 2 * gpuLDPCSoftColumnWords
	gpuLDPCSoftMessageWords      = gpuLDPCMaxBlocks * gpuLDPCSoftMaxEdges
	gpuLDPCSoftIndirectWords     = 9
)

const (
	gpuLDPCSoftReliabilityIndirectOffset = iota * 3 * 4
	gpuLDPCSoftGraphIndirectOffset
	gpuLDPCSoftCorrectionIndirectOffset
)

// gpuLDPCSoftRetainedBytes is one bounded column-to-edge view for each parity
// shape, one reliability word per resident codeword bit, one signed message per
// edge, and the three indirect dispatch records. The column view is built on the
// device only when hard correction fails.
const gpuLDPCSoftRetainedBytes = (gpuLDPCSoftColumnBufferWords +
	gpuLDPCMaxBlocks*gpuLDPCMaxSub + gpuLDPCSoftMessageWords +
	gpuLDPCSoftIndirectWords) * 4

type gpuLDPCSoftPlan struct {
	edges     int
	tailEdges int
}

func gpuLDPCSoftPlanOf(plan gpuLDPCPlan) (gpuLDPCSoftPlan, error) {
	edges, err := gpuLDPCSoftEdgesOf(
		plan.rows, plan.rowDegree, plan.height, plan.length,
	)
	if err != nil {
		return gpuLDPCSoftPlan{}, err
	}
	soft := gpuLDPCSoftPlan{edges: edges}
	if plan.tailLength != 0 {
		soft.tailEdges, err = gpuLDPCSoftEdgesOf(
			plan.tailRows, plan.tailRowDegree, plan.tailHeight, plan.tailLength,
		)
		if err != nil {
			return gpuLDPCSoftPlan{}, err
		}
	}
	messages := (plan.blocks - 1) * edges
	if plan.tailLength != 0 {
		messages += soft.tailEdges
	} else {
		messages += edges
	}
	if messages > gpuLDPCSoftMessageWords {
		return gpuLDPCSoftPlan{}, fmt.Errorf("jabcode: GPU soft LDPC messages exceed resident storage")
	}
	return soft, nil
}

func gpuLDPCSoftEdgesOf(rows []uint32, degree, height, length int) (int, error) {
	edges := degree * height
	if degree <= 0 || height <= 0 || length <= 0 || length > gpuLDPCMaxSub ||
		height > length || edges > len(rows) || edges > gpuLDPCSoftMaxEdges {
		return 0, fmt.Errorf("jabcode: GPU soft LDPC graph shape is out of range")
	}
	columnCounts := make([]uint8, length)
	for _, column := range rows[:edges] {
		if column == ecc.ParityRowPad || column >= uint32(length) {
			return 0, fmt.Errorf("jabcode: GPU soft LDPC graph is not compact")
		}
		columnCounts[column]++
		if columnCounts[column] > gpuLDPCSoftMaxColumnDegree {
			return 0, fmt.Errorf(
				"jabcode: GPU soft LDPC column degree exceeds %d", gpuLDPCSoftMaxColumnDegree)
		}
	}
	return edges, nil
}
