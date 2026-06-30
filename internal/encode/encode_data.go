package encode

import (
	"errors"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// The mode-optimizer state space: indices 0..6 are the base encoding modes (the
// spec.Mode* values, "latch" states) and 7..13 the corresponding "shift" states.
const (
	numEncodingModes = 6  // base modes covered by tables.EncodingTable (byte handled apart)
	numModes         = 14 // latch + shift states
)

var errEncode = errors.New("jabcode: encoding data failed")

func iabs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// writeBits writes the low `length` bits of dec into bits[pos:pos+length],
// most-significant bit first. Each output byte holds a single 0/1 bit.
func writeBits(bits []byte, dec, pos, length int) {
	// Ports convert_dec_to_bin in encoder.c.
	if dec < 0 {
		dec += 256
	}
	for j := range length {
		bits[pos+length-1-j] = byte(dec & 1)
		dec >>= 1
	}
}

// analyzeInputData finds the minimum-length encoding-mode sequence for the input
// via dynamic programming over the 14 latch/shift states, returning the mode
// sequence and the total encoded bit length.
//
// curr[s][m] is the fewest bits to encode the first s characters ending in mode
// m; prev[s][m] records the predecessor mode for backtracking. Modes 7..13 are
// single-character "shifts" that revert afterwards, tracked via switchMode.
func analyzeInputData(input []byte) (seq []int, encodedLength int) {
	// Ports analyzeInputData in encoder.c.
	n := len(input)

	curr := make([][numModes]int, n+2)
	prev := make([][numModes]int, 2*n+2)
	for s := range prev {
		for m := range numModes {
			prev[s][m] = tables.EncMax / 2
		}
	}
	var switchMode, tempSwitchMode [2 * numModes]int
	for i := range switchMode {
		switchMode[i] = tables.EncMax / 2
		tempSwitchMode[i] = tables.EncMax / 2
	}

	// Start in upper-case mode with no predecessor.
	for m := range 7 {
		curr[0][m] = tables.EncMax
		curr[0][m+7] = tables.EncMax
		prev[0][m] = tables.EncMax
		prev[0][m+7] = tables.EncMax
	}
	curr[0][0] = 0

	jumpNext := false // current char pairs with the next (CR/LF, "x ")
	confirm := false
	isShift := false
	nbChar := 0
	endOfLoop := n
	prevModeIndex := 0
	lastStep := 0

	for i := 0; i < endOfLoop; i++ {
		tmp := int(input[nbChar])
		tmp1 := 0
		if nbChar+1 < n {
			tmp1 = int(input[nbChar+1])
		}
		step := i + 1
		lastStep = step

		// Cost to encode this character in each base mode.
		for j := range numEncodingModes {
			v := tables.EncodingTable[tmp][j]
			switch {
			case v > -1 && v < 64:
				curr[step][j] = tables.CharacterSize[j]
				curr[step][j+7] = tables.CharacterSize[j]
			case (v == -18 && tmp1 == 10) || (v < -18 && tmp1 == 32):
				curr[step][j] = tables.CharacterSize[j]
				curr[step][j+7] = tables.CharacterSize[j]
				jumpNext = true
			default:
				curr[step][j] = tables.EncMax
				curr[step][j+7] = tables.EncMax
			}
		}
		curr[step][spec.ModeByte] = tables.CharacterSize[spec.ModeByte] // byte mode always works
		curr[step][spec.ModeByte+7] = tables.CharacterSize[spec.ModeByte]

		isShift = false
		for j := range numModes {
			prev2 := -1
			length := curr[step][j] + curr[step-1][j] + tables.LatchShiftTo[j][j]
			prev[step][j] = j
			for k := range numModes {
				if (length >= curr[step][j]+curr[step-1][k]+tables.LatchShiftTo[k][j] && k < 13) || (k == 13 && prev2 == j) {
					length = curr[step][j] + curr[step-1][k] + tables.LatchShiftTo[k][j]
					if tempSwitchMode[2*k] == k {
						prev[step][j] = tempSwitchMode[2*k+1]
					} else {
						prev[step][j] = k
					}
					if k == 13 && prev2 == j {
						prev2 = -1
					}
				}
			}
			curr[step][j] = length

			// Shift states encode one character then revert; fold their cost back
			// into the latch state they were invoked from.
			if j > 6 {
				pm := prev[step][j]
				switch {
				case (curr[step][pm] > length || (jumpNext && curr[step][pm]+tables.CharacterSize[pm%7] > length)) && j != 13:
					index := pm
					for loop := 1; index > 6 && step-loop >= 0; loop++ {
						index = prev[step-loop][index]
					}
					curr[step][index] = length
					prev[step+1][index] = j
					switchMode[2*index] = index
					switchMode[2*index+1] = j
					isShift = true
					if jumpNext && j == 11 {
						confirm = true
						prevModeIndex = index
					}
				case (curr[step][pm] > length || (jumpNext && curr[step][pm]+tables.CharacterSize[pm%7] > length)) && j == 13:
					curr[step][pm] = length
					prev[step+1][pm] = j
					switchMode[2*pm] = pm
					switchMode[2*pm+1] = j
					isShift = true
				}
				if j != 13 {
					curr[step][j] = tables.EncMax
				} else {
					prev2 = prev[step][j]
				}
			}
		}

		for j := range 2 * numModes {
			tempSwitchMode[j] = switchMode[j]
			switchMode[j] = tables.EncMax / 2
		}

		if jumpNext && confirm {
			for j := 0; j <= 2*numEncodingModes+1; j++ {
				if j != prevModeIndex {
					curr[step][j] = tables.EncMax
				}
			}
			nbChar++
			endOfLoop--
		}
		jumpNext = false
		confirm = false
		nbChar++
	}

	// Pick the cheapest final state.
	encodeSeqLength := tables.EncMax
	currentMode := 0
	for j := 0; j <= 2*numEncodingModes+1; j++ {
		if encodeSeqLength > curr[lastStep][j] {
			encodeSeqLength = curr[lastStep][j]
			currentMode = j
		}
	}
	if currentMode > 6 {
		isShift = true
	}
	if isShift && tempSwitchMode[2*currentMode+1] < 14 {
		currentMode = tempSwitchMode[2*currentMode+1]
	}

	// Backtrack to recover the mode per character, accounting for the extra
	// length headers when byte mode runs longer than 15 (and 8207) characters.
	seq = make([]int, lastStep+1+b2i(isShift))
	seq[lastStep] = currentMode
	counter := 0
	modeswitch := 0
	for i := lastStep; i > 0; i-- {
		if seq[i] == 13 || seq[i] == 6 {
			counter++
		} else if counter > 15 {
			encodeSeqLength += 13
			if counter > 8207 { // 2^13 + 15
				switch {
				case seq[i] == 0 || seq[i] == 1 || seq[i] == 7 || seq[i] == 8:
					modeswitch = 11
				case seq[i] == 2 || seq[i] == 9:
					modeswitch = 10
				case seq[i] == 5 || seq[i] == 12:
					modeswitch = 12
				}
				encodeSeqLength += byteRunOverhead(counter, modeswitch)
			}
			counter = 0
		}

		switch {
		case seq[i] < 14 && i-1 != 0:
			seq[i-1] = prev[i][seq[i]]
		case i-1 == 0:
			seq[i-1] = 0
			if counter > 15 {
				encodeSeqLength += 13
				if counter > 8207 {
					modeswitch = 11
					encodeSeqLength += byteRunOverhead(counter, modeswitch)
				}
				counter = 0
			}
		default:
			return nil, 0
		}
	}
	return seq, encodeSeqLength
}

// byteRunOverhead returns the extra encoded length for a byte-mode run longer
// than 8207 characters, which must be split into repeated length-prefixed
// chunks (the counter>8207 accounting in analyzeInputData).
func byteRunOverhead(counter, modeswitch int) int {
	remain := counter / 8207
	residual := counter % 8207
	extra := remain * modeswitch
	if residual < 16 {
		extra += (remain - 1) * 13
	} else {
		extra += remain * 13
	}
	if residual == 0 {
		extra -= modeswitch
	}
	return extra
}

// encodeData encodes the input into a bit-per-byte bitstream of length
// encodedLength, following the optimal mode sequence from analyzeInputData.
// It may mutate seq (shift-back bookkeeping).
func encodeData(input []byte, encodedLength int, seq []int) ([]byte, error) {
	// Ports encodeData in encoder.c.
	encoded := make([]byte, encodedLength)

	counter := 0
	shiftBack := false
	position := 0
	current := 0 // index into input
	endOfLoop := len(input)
	byteOffset := 0
	byteCounter := 0
	factor := 1

	for i := 0; i < endOfLoop; i++ {
		tmp := int(input[current])
		if position >= encodedLength {
			return nil, errEncode
		}

		// Emit a mode switch if the mode changed.
		if seq[counter] != seq[counter+1] {
			length := tables.LatchShiftTo[seq[counter]][seq[counter+1]]
			if seq[counter+1] == 6 || seq[counter+1] == 13 {
				length -= 4
			}
			if length >= tables.EncMax {
				return nil, errEncode
			}
			writeBits(encoded, tables.ModeSwitch[seq[counter]][seq[counter+1]], position, length)
			position += tables.LatchShiftTo[seq[counter]][seq[counter+1]]
			if seq[counter+1] == 6 || seq[counter+1] == 13 {
				position -= 4
			}
			if (seq[counter+1] > 6 && seq[counter+1] <= 13) || (seq[counter+1] == 13 && seq[counter+2] != 13) {
				shiftBack = true // revert after this character
			}
		}

		if seq[counter+1]%7 != spec.ModeByte {
			et := tables.EncodingTable[tmp][seq[counter+1]%7]
			switch {
			case et > -1 && tables.CharacterSize[seq[counter+1]%7] < tables.EncMax:
				writeBits(encoded, et, position, tables.CharacterSize[seq[counter+1]%7])
				position += tables.CharacterSize[seq[counter+1]%7]
				counter++
			case et < -1:
				tmp1 := int(input[current+1])
				var decimal int
				switch {
				case (tmp == 44 || tmp == 46 || tmp == 58) && tmp1 == 32, tmp == 13 && tmp1 == 10:
					decimal = iabs(et)
				case tmp == 13 && tmp1 != 10:
					decimal = 18
				default:
					return nil, errEncode
				}
				if tables.CharacterSize[seq[counter+1]%7] < tables.EncMax {
					writeBits(encoded, decimal, position, tables.CharacterSize[seq[counter+1]%7])
				}
				position += tables.CharacterSize[seq[counter+1]%7]
				counter++
				endOfLoop--
				current++
			default:
				return nil, errEncode
			}
		} else {
			// Byte mode: write the run length, then the raw byte.
			if seq[counter] != seq[counter+1] {
				byteCounter = 0
				for bl := counter + 1; bl <= endOfLoop; bl++ {
					if seq[bl] == 6 || seq[bl] == 13 {
						byteCounter++
					} else {
						break
					}
				}
				writeBits(encoded, cap15(byteCounter), position, 4)
				position += 4
				if byteCounter > 15 {
					if byteCounter <= 8207 {
						writeBits(encoded, byteCounter-15-1, position, 13)
					} else {
						writeBits(encoded, 8191, position, 13)
					}
					position += 13
				}
				byteOffset = byteCounter
			}
			if byteOffset-byteCounter == factor*8207 { // run exceeds 2^13+15: re-enter byte mode
				switch seq[counter-(byteOffset-byteCounter)] {
				case 0, 7, 1, 8:
					writeBits(encoded, 124, position, 7) // shift upper -> byte
					position += 7
				case 2, 9:
					writeBits(encoded, 60, position, 5) // shift numeric -> byte
					position += 5
				case 5, 12:
					writeBits(encoded, 252, position, 8) // shift alphanumeric -> byte
					position += 8
				}
				writeBits(encoded, cap15(byteCounter), position, 4)
				position += 4
				if byteCounter > 15 {
					if byteCounter <= 8207 {
						writeBits(encoded, byteCounter-15-1, position, 13)
					} else {
						writeBits(encoded, 8191, position, 13)
					}
					position += 13
				}
				factor++
			}
			if tables.CharacterSize[seq[counter+1]%7] >= tables.EncMax {
				return nil, errEncode
			}
			writeBits(encoded, tmp, position, tables.CharacterSize[seq[counter+1]%7])
			position += tables.CharacterSize[seq[counter+1]%7]
			counter++
			byteCounter--
		}

		// Revert a shift back to the mode it was invoked from.
		if shiftBack && byteCounter == 0 {
			if byteOffset == 0 {
				seq[counter] = seq[counter-1]
			} else {
				seq[counter] = seq[counter-byteOffset]
			}
			shiftBack = false
			byteOffset = 0
		}
		current++
	}
	return encoded, nil
}

// cap15 returns n, or 0 when n exceeds 15 (the 4-bit byte-run length field that
// signals an extended 13-bit length follows).
func cap15(n int) int {
	if n > 15 {
		return 0
	}
	return n
}
