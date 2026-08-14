//go:build !js

package detect

import (
	"fmt"
	"math/rand/v2"
	"regexp"
	"strconv"
	"testing"
)

// matrixModel mirrors the elimination the resident matrix builder runs, at
// exactly the level the blocked restructure changed: which rows are combined, in
// what order, and what column arrangement the sweep records on the way. The
// device kernel is not under test here and cannot be - only the claim that
// visiting the pivots in panels is the same elimination as visiting them one at
// a time, which is a property of the traversal rather than of any adapter.
type matrixModel struct {
	height, stride int
	dense          []uint32
	arrangement    []int
	processed      []bool
	zeroLines      []int
	swaps          [][2]int
}

func newMatrixModel(height, stride int, dense []uint32) *matrixModel {
	return &matrixModel{
		height:      height,
		stride:      stride,
		dense:       append([]uint32(nil), dense...),
		arrangement: make([]int, height),
		processed:   make([]bool, height),
	}
}

func (m *matrixModel) bit(row, column int) bool {
	return m.dense[row*m.stride+column/32]>>(column%32)&1 != 0
}

func (m *matrixModel) firstSet(row int) int {
	for word := range m.stride {
		if value := m.dense[row*m.stride+word]; value != 0 {
			for bit := range 32 {
				if value>>bit&1 != 0 {
					return word*32 + bit
				}
			}
		}
	}
	return -1
}

func (m *matrixModel) xor(target, source int) {
	for word := range m.stride {
		m.dense[target*m.stride+word] ^= m.dense[source*m.stride+word]
	}
}

// record is the per-pivot bookkeeping both traversals share.
func (m *matrixModel) record(row, pivot int) {
	if pivot < m.height {
		m.processed[pivot] = true
	}
	if pivot < len(m.arrangement) {
		m.arrangement[pivot] = row
	}
	if pivot >= m.height {
		m.swaps = append(m.swaps, [2]int{pivot, 0})
	}
}

// sweep is the elimination as one sequential pass over the pivots.
func (m *matrixModel) sweep() {
	for row := range m.height {
		pivot := m.firstSet(row)
		if pivot < 0 {
			m.zeroLines = append(m.zeroLines, row)
			continue
		}
		m.record(row, pivot)
		for target := range m.height {
			if target != row && m.bit(target, pivot) {
				m.xor(target, row)
			}
		}
	}
}

// blocked is the elimination as the kernel now runs it: a panel of pivots
// reduces its own rows, then every other row takes the finished panel rows its
// pre-panel bits at those pivot columns select.
func (m *matrixModel) blocked(panel int) {
	for base := 0; base < m.height; base += panel {
		span := min(panel, m.height-base)
		var columns, rows []int
		for index := range span {
			row := base + index
			pivot := m.firstSet(row)
			if pivot < 0 {
				m.zeroLines = append(m.zeroLines, row)
				continue
			}
			m.record(row, pivot)
			for target := base; target < base+span; target++ {
				if target != row && m.bit(target, pivot) {
					m.xor(target, row)
				}
			}
			columns = append(columns, pivot)
			rows = append(rows, row)
		}
		for target := range m.height {
			if target >= base && target < base+span {
				continue
			}
			selected := 0
			for slot, column := range columns {
				if m.bit(target, column) {
					selected |= 1 << slot
				}
			}
			for slot := range columns {
				if selected&(1<<slot) != 0 {
					m.xor(target, rows[slot])
				}
			}
		}
	}
}

// modelSource builds a Gallager-shaped dense matrix: every row carries wr ones,
// which is the input the elimination actually meets.
func modelSource(length, wc, wr int, seed uint64) (int, int, []uint32) {
	height := length / wr * wc
	stride := (length + 31) / 32
	dense := make([]uint32, height*stride)
	source := rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15))
	for row := range height {
		for range wr {
			column := source.IntN(length)
			dense[row*stride+column/32] |= 1 << (column % 32)
		}
	}
	return height, stride, dense
}

// TestGPULDPCMatrixBlockedEliminationMatchesSweep is the gate on the restructure
// itself. Hard LDPC has no payload integrity check, so a parity matrix that is
// merely plausible decodes into silent corruption; the elimination has to agree
// bit for bit, not approximately.
func TestGPULDPCMatrixBlockedEliminationMatchesSweep(t *testing.T) {
	shapes := []struct{ length, wc, wr int }{
		{2560, 4, 8},
		{2816, 5, 10},
		{1024, 3, 4},
		{330, 3, 11},
		{96, 3, 4},
		{64, 3, 8},
		// A single panel, and one bit under a panel boundary, so the last
		// partial panel and the no-trailing-work case are both covered.
		{32, 3, 4},
		{124, 3, 4},
	}
	for _, shape := range shapes {
		for seed := range uint64(6) {
			name := fmt.Sprintf("len%d_wc%d_wr%d_seed%d", shape.length, shape.wc, shape.wr, seed)
			t.Run(name, func(t *testing.T) {
				height, stride, dense := modelSource(shape.length, shape.wc, shape.wr, seed)
				sequential := newMatrixModel(height, stride, dense)
				sequential.sweep()
				panelled := newMatrixModel(height, stride, dense)
				panelled.blocked(gpuLDPCMatrixPanel)

				for at := range sequential.dense {
					if sequential.dense[at] != panelled.dense[at] {
						t.Fatalf("reduced matrix word %d: sweep %#08x, blocked %#08x",
							at, sequential.dense[at], panelled.dense[at])
					}
				}
				for at := range sequential.arrangement {
					if sequential.arrangement[at] != panelled.arrangement[at] {
						t.Fatalf("arrangement[%d]: sweep %d, blocked %d",
							at, sequential.arrangement[at], panelled.arrangement[at])
					}
				}
				for at := range sequential.processed {
					if sequential.processed[at] != panelled.processed[at] {
						t.Fatalf("processed[%d]: sweep %v, blocked %v",
							at, sequential.processed[at], panelled.processed[at])
					}
				}
				if len(sequential.zeroLines) != len(panelled.zeroLines) {
					t.Fatalf("zero lines: sweep %d, blocked %d",
						len(sequential.zeroLines), len(panelled.zeroLines))
				}
				for at := range sequential.zeroLines {
					if sequential.zeroLines[at] != panelled.zeroLines[at] {
						t.Fatalf("zero line %d: sweep %d, blocked %d",
							at, sequential.zeroLines[at], panelled.zeroLines[at])
					}
				}
				if len(sequential.swaps) != len(panelled.swaps) {
					t.Fatalf("swaps: sweep %d, blocked %d",
						len(sequential.swaps), len(panelled.swaps))
				}
				for at := range sequential.swaps {
					if sequential.swaps[at] != panelled.swaps[at] {
						t.Fatalf("swap %d: sweep %v, blocked %v",
							at, sequential.swaps[at], panelled.swaps[at])
					}
				}
			})
		}
	}
}

// TestGPULDPCMatrixConstantsMatchShader pins the few numbers the host duplicates
// from the shader. The apply phase reads its dispatch grid out of the workspace,
// so a drifted offset does not fail loudly: it launches whatever words happen to
// sit there.
func TestGPULDPCMatrixConstantsMatchShader(t *testing.T) {
	for _, want := range []struct {
		name  string
		value int
	}{
		{"PANEL", gpuLDPCMatrixPanel},
		{"MAX_SUB", gpuLDPCMaxSub},
		{"STATE_APPLY_DIMS", gpuLDPCMatrixStateApplyDims},
		{"STATE_PIVOT_COLUMN", gpuLDPCMatrixStateWords - 2*gpuLDPCMatrixPanel},
	} {
		pattern := regexp.MustCompile(`(?m)^const ` + want.name + `: u32 = (\d+)u;`)
		match := pattern.FindStringSubmatch(ldpcMatrixWGSL)
		if match == nil {
			t.Fatalf("shader declares no %s", want.name)
		}
		found, err := strconv.Atoi(match[1])
		if err != nil {
			t.Fatalf("%s: %v", want.name, err)
		}
		if found != want.value {
			t.Fatalf("%s: shader %d, host %d", want.name, found, want.value)
		}
	}
}
