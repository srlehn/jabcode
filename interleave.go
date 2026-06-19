package jabcode

// interleaveSeed is the fixed LCG seed used to derive the (de)interleaving
// permutation (INTERLEAVE_SEED in interleave.c).
const interleaveSeed = 226759

// interleaveData shuffles data in place using a deterministic back-to-front
// Fisher-Yates pass driven by the seeded generator (interleaveData in
// interleave.c).
func interleaveData(data []byte) {
	n := len(data)
	if n == 0 {
		return
	}
	i := 0
	for x := range lcgValues(interleaveSeed) {
		pos := randIndex(x, n-i)
		j := n - 1 - i
		data[j], data[pos] = data[pos], data[j]
		if i++; i == n {
			break
		}
	}
}

// deinterleaveData inverts interleaveData in place. It replays the interleaving
// permutation on an index array, then scatters the data back to its original
// positions (deinterleaveData in interleave.c).
func deinterleaveData(data []byte) {
	n := len(data)
	if n == 0 {
		return
	}
	index := make([]int, n)
	for i := range index {
		index[i] = i
	}
	i := 0
	for x := range lcgValues(interleaveSeed) {
		pos := randIndex(x, n-i)
		j := n - 1 - i
		index[j], index[pos] = index[pos], index[j]
		if i++; i == n {
			break
		}
	}
	tmp := make([]byte, n)
	copy(tmp, data)
	for i := range n {
		data[index[i]] = tmp[i]
	}
}
