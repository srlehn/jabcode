package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

// Decoder-only encoding-mode values. The base modes spec.ModeUpper..spec.ModeByte
// (0..6) are shared with the encoder.
const (
	modeNone = -1
	modeECI  = 7
	modeFNC1 = 8
)

// Decoding tables mapping mode values to output bytes.
var (
	decodingTableUpper        = []byte{32, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90}
	decodingTableLower        = []byte{32, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122}
	decodingTableNumeric      = []byte{32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 44, 46}
	decodingTablePunct        = []byte{33, 34, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 58, 59, 63, 64}
	decodingTableMixed        = []byte{35, 42, 43, 60, 61, 62, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126, 9, 10, 13, 0, 0, 0, 0, 164, 167, 196, 214, 220, 223, 228, 246, 252}
	decodingTableAlphanumeric = []byte{32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122}
)

// messageOutput applies the ISO transmitted-data protocol while interpreting
// the mode stream. The ISO profile models an ECI-capable reader: every
// transmission carries its Annex H identifier, every literal data backslash
// is doubled, and each ECI assignment is a backslash plus six decimal digits.
type messageOutput struct {
	profile      wire.Profile
	data         []byte
	dataCount    int
	leading      [2]byte
	fnc1Active   bool
	fnc1Modifier byte
}

func (o *messageOutput) appendData(values ...byte) {
	for _, value := range values {
		if o.dataCount < len(o.leading) {
			o.leading[o.dataCount] = value
		}
		o.dataCount++
		if o.profile == wire.ISO23634 && value == '\\' {
			o.data = append(o.data, '\\')
		}
		o.data = append(o.data, value)
	}
}

func (o *messageOutput) appendECI(assignment int) {
	o.data = append(o.data, '\\')
	var digits [6]byte
	for i := len(digits) - 1; i >= 0; i-- {
		digits[i] = byte('0' + assignment%10)
		assignment /= 10
	}
	o.data = append(o.data, digits[:]...)
}

func (o *messageOutput) fnc1() bool {
	if o.fnc1Active {
		o.appendData(29) // in-mode FNC1 is the GS1 field separator
		return true
	}
	switch {
	case o.dataCount == 0:
		o.fnc1Modifier = 4
	case o.dataCount == 1 && isASCIILetter(o.leading[0]):
		o.fnc1Modifier = 5
	case o.dataCount == 2 && isASCIIDigit(o.leading[0]) && isASCIIDigit(o.leading[1]):
		o.fnc1Modifier = 5
	default:
		return false
	}
	o.fnc1Active = true
	return true
}

func (o *messageOutput) eot() bool {
	if !o.fnc1Active {
		return false
	}
	o.fnc1Active = false
	return true
}

func (o *messageOutput) finish() ([]byte, bool) {
	if o.profile != wire.ISO23634 {
		return o.data, true
	}
	if o.fnc1Active {
		return nil, false
	}
	modifier := o.fnc1Modifier
	if modifier == 0 {
		modifier = 1
	}
	data := make([]byte, 0, len(o.data)+3)
	data = append(data, ']', 'j', '0'+modifier)
	data = append(data, o.data...)
	return data, true
}

func (o *messageOutput) fail() ([]byte, bool) {
	if o.profile == wire.ISO23634 {
		return nil, false
	}
	return o.finish()
}

func isASCIILetter(value byte) bool {
	return value >= 'A' && value <= 'Z' || value >= 'a' && value <= 'z'
}

func isASCIIDigit(value byte) bool { return value >= '0' && value <= '9' }

// readData reads up to length bits from data starting at start, MSB first,
// returning the value and the number of bits actually read.
func readData(data []byte, start, length int) (value, n int) {
	// Ports readData in decoder.c.
	i := start
	for ; i < start+length && i < len(data); i++ {
		value += int(data[i]) << (length - 1 - (i - start))
	}
	return value, i - start
}

// readECIAssignment decodes the variable-length assignment from ISO Table 19.
// The leading 0, 10 or 11 bits select an 8-, 16- or 22-bit field.
func readECIAssignment(bits []byte, start int) (assignment, next int, ok bool) {
	first, n := readData(bits, start, 1)
	if n != 1 {
		return 0, start, false
	}
	total, valueBits := 8, 7
	if first == 1 {
		second, n := readData(bits, start+1, 1)
		if n != 1 {
			return 0, start, false
		}
		if second == 0 {
			total, valueBits = 16, 14
		} else {
			total, valueBits = 22, 20
		}
	}
	encoded, n := readData(bits, start, total)
	if n != total {
		return 0, start, false
	}
	assignment = encoded & ((1 << valueBits) - 1)
	if assignment > 999999 {
		return 0, start, false
	}
	return assignment, start + total, true
}

// demaskSymbol removes the data mask from raw module values in place. Modules are
// visited in column-major order, matching readRawModuleData.
func demaskSymbol(data, dataMap []byte, size image.Point, maskType, colorNumber int) {
	// Ports demaskSymbol in mask.c.
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
			idx ^= spec.MaskValue(maskType, x, y) % colorNumber
			data[count] = byte(idx)
			count++
		}
	}
}

// DecodeData interprets the corrected bit stream into the decoded message,
// following the mode/latch/shift state machine.
func DecodeData(bits []byte) []byte {
	data, _ := DecodeDataProfile(bits, wire.CReference)
	return data
}

// DecodeDataProfile interprets a corrected bit stream under the selected wire
// profile. ok is false when an ISO stream is truncated, uses a reserved switch,
// or violates the FNC1 start/end protocol. C-reference mode preserves the
// reference decoder's partial-message behavior.
func DecodeDataProfile(bits []byte, profile wire.Profile) ([]byte, bool) {
	// Ports decodeData in decoder.c.
	output := messageOutput{profile: profile}
	mode := spec.ModeUpper
	preMode := modeNone
	index := 0

	for index < len(bits) {
		if mode == modeECI || mode == modeFNC1 || mode == modeNone {
			// The C reference enters sentinel ECI/FNC1 modes that it does not
			// interpret. The ISO path handles those controls inline and cannot
			// normally reach these values. None is always an error state.
			if profile == wire.ISO23634 {
				return output.fail()
			}
			return output.finish()
		}
		flag := false
		value := 0
		var n int
		if mode != spec.ModeByte {
			value, n = readData(bits, index, tables.CharacterSize[mode])
			if n < tables.CharacterSize[mode] {
				if profile == wire.ISO23634 {
					return output.fail()
				}
				break
			}
			index += tables.CharacterSize[mode]
		}

		switch mode {
		case spec.ModeUpper:
			if value <= 26 {
				output.appendData(decodingTableUpper[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else {
				switch value {
				case 27:
					mode, preMode = spec.ModePunct, spec.ModeUpper
				case 28:
					mode, preMode = spec.ModeLower, modeNone
				case 29:
					mode, preMode = spec.ModeNumeric, modeNone
				case 30:
					mode, preMode = spec.ModeAlphanumeric, modeNone
				case 31:
					value, n = readData(bits, index, 2)
					if n < 2 {
						if profile == wire.ISO23634 {
							return output.fail()
						}
						flag = true
						break
					}
					index += 2
					switch value {
					case 0:
						mode, preMode = spec.ModeByte, spec.ModeUpper
					case 1:
						mode, preMode = spec.ModeMixed, spec.ModeUpper
					case 2:
						if profile != wire.ISO23634 {
							mode, preMode = modeECI, modeNone
							break
						}
						assignment, next, ok := readECIAssignment(bits, index)
						if !ok {
							return output.fail()
						}
						output.appendECI(assignment)
						index = next
						mode = spec.ModeUpper
					case 3:
						if profile != wire.ISO23634 {
							flag = true // end of message in the C reference
							break
						}
						additional, n := readData(bits, index, 3)
						if n != 3 {
							return output.fail()
						}
						index += 3
						switch additional {
						case 0:
							// ISO/IEC 15434 has its own start/end transmission
							// protocol and is not part of the ECI/FNC1 path.
							return output.fail()
						case 1:
							output.appendData([]byte("https://")...)
						case 2:
							output.appendData([]byte("http://")...)
						case 3:
							output.appendData([]byte("www.")...)
						case 4:
							if !output.fnc1() {
								return output.fail()
							}
						case 5:
							if !output.eot() {
								return output.fail()
							}
						case 6, 7:
							return output.fail()
						}
						if additional >= 1 && additional <= 3 && preMode != modeNone {
							mode = preMode
						} else {
							mode, preMode = spec.ModeUpper, modeNone
						}
					}
				default:
					return output.fail()
				}
			}
		case spec.ModeLower:
			if value <= 26 {
				output.appendData(decodingTableLower[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else {
				switch value {
				case 27:
					mode, preMode = spec.ModePunct, spec.ModeLower
				case 28:
					mode, preMode = spec.ModeUpper, spec.ModeLower
				case 29:
					mode, preMode = spec.ModeNumeric, modeNone
				case 30:
					mode, preMode = spec.ModeAlphanumeric, modeNone
				case 31:
					value, n = readData(bits, index, 2)
					if n < 2 {
						if profile == wire.ISO23634 {
							return output.fail()
						}
						flag = true
						break
					}
					index += 2
					switch value {
					case 0:
						mode, preMode = spec.ModeByte, spec.ModeLower
					case 1:
						mode, preMode = spec.ModeMixed, spec.ModeLower
					case 2:
						mode, preMode = spec.ModeUpper, modeNone
					case 3:
						if profile == wire.ISO23634 {
							mode, preMode = spec.ModeNumeric, spec.ModeLower
						} else {
							mode, preMode = modeFNC1, modeNone
						}
					}
				default:
					return output.fail()
				}
			}
		case spec.ModeNumeric:
			if value <= 12 {
				output.appendData(decodingTableNumeric[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else {
				switch value {
				case 13:
					mode, preMode = spec.ModePunct, spec.ModeNumeric
				case 14:
					mode, preMode = spec.ModeUpper, modeNone
				case 15:
					value, n = readData(bits, index, 2)
					if n < 2 {
						if profile == wire.ISO23634 {
							return output.fail()
						}
						flag = true
						break
					}
					index += 2
					switch value {
					case 0:
						mode, preMode = spec.ModeByte, spec.ModeNumeric
					case 1:
						mode, preMode = spec.ModeMixed, spec.ModeNumeric
					case 2:
						mode, preMode = spec.ModeUpper, spec.ModeNumeric
					case 3:
						mode, preMode = spec.ModeLower, modeNone
					}
				default:
					return output.fail()
				}
			}
		case spec.ModePunct:
			if value >= 0 && value <= 15 {
				output.appendData(decodingTablePunct[value])
				mode = preMode
			} else {
				return output.fail()
			}
		case spec.ModeMixed:
			if value >= 0 && value <= 31 {
				switch value {
				case 19:
					output.appendData(10, 13)
				case 20:
					output.appendData(44, 32)
				case 21:
					output.appendData(46, 32)
				case 22:
					output.appendData(58, 32)
				default:
					output.appendData(decodingTableMixed[value])
				}
				mode = preMode
			} else {
				return output.fail()
			}
		case spec.ModeAlphanumeric:
			if value <= 62 {
				output.appendData(decodingTableAlphanumeric[value])
				if preMode != modeNone {
					mode = preMode
				}
			} else if value == 63 {
				value, n = readData(bits, index, 2)
				if n < 2 {
					if profile == wire.ISO23634 {
						return output.fail()
					}
					flag = true
					break
				}
				index += 2
				switch value {
				case 0:
					mode, preMode = spec.ModeByte, spec.ModeAlphanumeric
				case 1:
					mode, preMode = spec.ModeMixed, spec.ModeAlphanumeric
				case 2:
					mode, preMode = spec.ModePunct, spec.ModeAlphanumeric
				case 3:
					mode, preMode = spec.ModeUpper, modeNone
				}
			} else {
				return output.fail()
			}
		case spec.ModeByte:
			value, n = readData(bits, index, 4)
			if n < 4 {
				return output.fail()
			}
			index += 4
			if value == 0 { // length encoded in the next 13 bits
				value, n = readData(bits, index, 13)
				if n < 13 {
					return output.fail()
				}
				value += 15 + 1
				index += 13
			}
			byteLength := value
			for range byteLength {
				value, n = readData(bits, index, 8)
				if n < 8 {
					return output.fail()
				}
				index += 8
				output.appendData(byte(value))
			}
			mode = preMode
		}
		if flag {
			break
		}
	}
	return output.finish()
}
