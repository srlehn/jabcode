package ecc

// bitMatrix is a dense matrix over GF(2): every entry is a single bit. Rows are
// stored contiguously as 64-bit words. The packing is purely an internal
// efficiency detail — all access is by logical (row, column), so results are
// independent of word size.
type bitMatrix struct {
	rows, cols int
	stride     int // words per row
	w          []uint64
}

func newBitMatrix(rows, cols int) *bitMatrix {
	stride := (cols + 63) / 64
	return &bitMatrix{rows: rows, cols: cols, stride: stride, w: make([]uint64, rows*stride)}
}

func (m *bitMatrix) get(r, c int) bool {
	return m.w[r*m.stride+c/64]>>(uint(c)%64)&1 != 0
}

func (m *bitMatrix) set(r, c int) {
	m.w[r*m.stride+c/64] |= 1 << (uint(c) % 64)
}

func (m *bitMatrix) toggle(r, c int) {
	m.w[r*m.stride+c/64] ^= 1 << (uint(c) % 64)
}

// xorRow adds (XOR) row src into row dst, i.e. dst <- dst + src over GF(2).
func (m *bitMatrix) xorRow(dst, src int) {
	d := m.w[dst*m.stride : dst*m.stride+m.stride]
	s := m.w[src*m.stride : src*m.stride+m.stride]
	for k := range d {
		d[k] ^= s[k]
	}
}

// swapCols exchanges columns a and b in every row.
func (m *bitMatrix) swapCols(a, b int) {
	for r := 0; r < m.rows; r++ {
		if m.get(r, a) != m.get(r, b) {
			m.toggle(r, a)
			m.toggle(r, b)
		}
	}
}

// firstSetCol returns the column of the first set bit in row r, or -1 if the
// row is all zero.
func (m *bitMatrix) firstSetCol(r int) int {
	for c := 0; c < m.cols; c++ {
		if m.get(r, c) {
			return c
		}
	}
	return -1
}

func (m *bitMatrix) clone() *bitMatrix {
	n := &bitMatrix{rows: m.rows, cols: m.cols, stride: m.stride, w: make([]uint64, len(m.w))}
	copy(n.w, m.w)
	return n
}

// copyRowFrom copies row src of o into row dst of m; both must have equal width.
func (m *bitMatrix) copyRowFrom(dst int, o *bitMatrix, src int) {
	copy(m.w[dst*m.stride:dst*m.stride+m.stride], o.w[src*o.stride:src*o.stride+o.stride])
}
