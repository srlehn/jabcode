package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/tables"
)

// Encoding-mode values beyond the base modes (jab_encode_mode in decoder.h).
// modeUpper..modeByte (0..6) are defined in encode_data.go.
const (
	modeNone = -1
	modeECI  = 7
	modeFNC1 = 8
)

// metadata holds a decoded symbol's parameters (jab_metadata).
type metadata struct {
	defaultMode    bool
	Nc             int
	maskType       int
	dockedPosition int
	sideVersion    image.Point
	ecl            image.Point // (wc, wr)
}

// decodedSymbol holds a decoded symbol (jab_decoded_symbol).
type decodedSymbol struct {
	index            int
	hostIndex        int
	hostPosition     int
	sideSize         image.Point
	moduleSize       float64
	patternPositions [4]pointF
	meta             metadata
	secondaryMeta    [4]metadata
	palette          []byte
	data             []byte
}

// Decoding tables mapping mode values to output bytes (decoder.h).
var (
	decodingTableUpper        = []byte{32, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90}
	decodingTableLower        = []byte{32, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122}
	decodingTableNumeric      = []byte{32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 44, 46}
	decodingTablePunct        = []byte{33, 34, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 58, 59, 63, 64}
	decodingTableMixed        = []byte{35, 42, 43, 60, 61, 62, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126, 9, 10, 13, 0, 0, 0, 0, 164, 167, 196, 214, 220, 223, 228, 246, 252}
	decodingTableAlphanumeric = []byte{32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122}
)

// readData reads up to length bits from data starting at start, MSB first,
// returning the value and the number of bits actually read (readData in decoder.c).
func readData(data []byte, start, length int) (value, n int) {
	i := start
	for ; i < start+length && i < len(data); i++ {
		value += int(data[i]) << (length - 1 - (i - start))
	}
	return value, i - start
}

// demaskSymbol removes the data mask from raw module values in place
// (demaskSymbol in mask.c). Modules are visited in column-major order, matching
// readRawModuleData.
func demaskSymbol(data, dataMap []byte, size image.Point, maskType, colorNumber int) {
	w, h := size.X, size.Y
	count := 0
	for x := range w {
		for y := range h {
			if dataMap[y*w+x] != 0 {
				continue
			}
			if count > len(data)-1 {
				return
			}
			idx := int(data[count])
			idx ^= maskValue(maskType, x, y) % colorNumber
			data[count] = byte(idx)
			count++
		}
	}
}

// decodeData interprets the corrected bit stream into the decoded message,
// following the mode/latch/shift state machine (decodeData in decoder.c).
func decodeData(bits []byte) []byte {
	var out []byte
	mode := modeUpper
	preMode := modeNone
	index := 0

	for index < len(bits) {
		flag := false
		value := 0
		var n int
		if mode != modeByte {
			value, n = readData(bits, index, tables.CharacterSize[mode])
			if n < tables.CharacterSize[mode] {
				break
			}
			index += tables.CharacterSize[mode]
		}

		switch mode {
		case modeUpper:
			if value <= 26 {
				out = append(out, decodingTableUpper[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else {
				switch value {
				case 27:
					mode, preMode = modePunct, modeUpper
				case 28:
					mode, preMode = modeLower, modeNone
				case 29:
					mode, preMode = modeNumeric, modeNone
				case 30:
					mode, preMode = modeAlphanumeric, modeNone
				case 31:
					value, n = readData(bits, index, 2)
					if n < 2 {
						flag = true
						break
					}
					index += 2
					switch value {
					case 0:
						mode, preMode = modeByte, modeUpper
					case 1:
						mode, preMode = modeMixed, modeUpper
					case 2:
						mode, preMode = modeECI, modeNone
					case 3:
						flag = true // end of message
					}
				default:
					return out
				}
			}
		case modeLower:
			if value <= 26 {
				out = append(out, decodingTableLower[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else {
				switch value {
				case 27:
					mode, preMode = modePunct, modeLower
				case 28:
					mode, preMode = modeUpper, modeLower
				case 29:
					mode, preMode = modeNumeric, modeNone
				case 30:
					mode, preMode = modeAlphanumeric, modeNone
				case 31:
					value, n = readData(bits, index, 2)
					if n < 2 {
						flag = true
						break
					}
					index += 2
					switch value {
					case 0:
						mode, preMode = modeByte, modeLower
					case 1:
						mode, preMode = modeMixed, modeLower
					case 2:
						mode, preMode = modeUpper, modeNone
					case 3:
						mode, preMode = modeFNC1, modeNone
					}
				default:
					return out
				}
			}
		case modeNumeric:
			if value <= 12 {
				out = append(out, decodingTableNumeric[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else {
				switch value {
				case 13:
					mode, preMode = modePunct, modeNumeric
				case 14:
					mode, preMode = modeUpper, modeNone
				case 15:
					value, n = readData(bits, index, 2)
					if n < 2 {
						flag = true
						break
					}
					index += 2
					switch value {
					case 0:
						mode, preMode = modeByte, modeNumeric
					case 1:
						mode, preMode = modeMixed, modeNumeric
					case 2:
						mode, preMode = modeUpper, modeNumeric
					case 3:
						mode, preMode = modeLower, modeNone
					}
				default:
					return out
				}
			}
		case modePunct:
			if value >= 0 && value <= 15 {
				out = append(out, decodingTablePunct[value])
				mode = preMode
			} else {
				return out
			}
		case modeMixed:
			if value >= 0 && value <= 31 {
				switch value {
				case 19:
					out = append(out, 10, 13)
				case 20:
					out = append(out, 44, 32)
				case 21:
					out = append(out, 46, 32)
				case 22:
					out = append(out, 58, 32)
				default:
					out = append(out, decodingTableMixed[value])
				}
				mode = preMode
			} else {
				return out
			}
		case modeAlphanumeric:
			if value <= 62 {
				out = append(out, decodingTableAlphanumeric[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else if value == 63 {
				value, n = readData(bits, index, 2)
				if n < 2 {
					flag = true
					break
				}
				index += 2
				switch value {
				case 0:
					mode, preMode = modeByte, modeAlphanumeric
				case 1:
					mode, preMode = modeMixed, modeAlphanumeric
				case 2:
					mode, preMode = modePunct, modeAlphanumeric
				case 3:
					mode, preMode = modeUpper, modeNone
				}
			} else {
				return out
			}
		case modeByte:
			value, n = readData(bits, index, 4)
			if n < 4 {
				return out
			}
			index += 4
			if value == 0 { // length encoded in the next 13 bits
				value, n = readData(bits, index, 13)
				if n < 13 {
					return out
				}
				value += 15 + 1
				index += 13
			}
			byteLength := value
			for range byteLength {
				value, n = readData(bits, index, 8)
				if n < 8 {
					return out
				}
				index += 8
				out = append(out, byte(value))
			}
			mode = preMode
		case modeECI, modeFNC1, modeNone:
			// ECI and FNC1 are left unimplemented by the reference too (decodeData
			// in decoder.c); None is an error. All three advance past the end to
			// stop decoding.
			index += len(bits)
		}
		if flag {
			break
		}
	}
	return out
}
