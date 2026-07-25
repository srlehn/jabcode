package core

// The five-pixel majority passes run over binarizer output, whose bytes are
// only ever 0 or 255. On such bytes a bitwise operation is a per-byte boolean,
// so eight pixels vote at once inside a single word and the byte-at-a-time
// running counter the passes used to keep disappears along with its
// read-modify-write. The vector kernels already relied on this same property;
// the scalar paths now state it too, which also removes a divergence where the
// two would have disagreed on input that is neither 0 nor 255.
const majorityWord = 8

// majority5Mask votes five 0x00/0xFF byte masks per byte lane, returning 0xFF
// in each lane where at least three of the five inputs are set.
//
// The count is formed by two full adders rather than the ten three-of-five
// conjunctions the vector kernel spells out, which is the same result in about
// half the operations. With s the low bit of the count and c1, c2 the two
// carries, the count is s + 2*(c1 + c2), so it reaches three exactly when both
// carries fire, or when one fires and the low bit is set.
func majority5Mask(a, b, c, d, e uint64) uint64 {
	ab := a ^ b
	s1 := ab ^ c
	c1 := (a & b) | (ab & c)
	de := s1 ^ d
	s2 := de ^ e
	c2 := (s1 & d) | (de & e)
	return (c1 & c2) | ((c1 | c2) & s2)
}
