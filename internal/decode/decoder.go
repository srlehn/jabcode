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

// MessageControlKind identifies a non-data control encountered while decoding
// a message mode stream.
type MessageControlKind uint8

const (
	MessageControlECI MessageControlKind = iota + 1
	MessageControlFNC1Start
	MessageControlFNC1Separator
	MessageControlFNC1End
	MessageControlISO15434Start
	MessageControlISO15434End
)

// MessageControl records a control at an offset in Message.Data. Assignment is
// set only for MessageControlECI.
type MessageControl struct {
	Kind       MessageControlKind
	Offset     int
	Assignment int
}

// Message contains the decoded application data and its standards-facing
// reader transmission. Controls remain separate from Data so ECI and structured
// message semantics are not confused with literal payload bytes.
type Message struct {
	Data               []byte
	ReaderTransmission []byte
	Controls           []MessageControl
	Variant            wire.Variant
}

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
// the mode stream. The ISO variant models an ECI-capable reader: every
// transmission carries its Annex H identifier, every literal data backslash
// is doubled, and each ECI assignment is a backslash plus six decimal digits.
type messageOutput struct {
	variant             wire.Variant
	raw                 []byte
	transmission        []byte
	controls            []MessageControl
	dataCount           int
	leading             [2]byte
	fnc1Active          bool
	fnc1Modifier        byte
	iso15434Active      bool
	iso15434Used        bool
	iso15434Format      [2]byte
	iso15434FormatCount int
}

func (o *messageOutput) appendData(values ...byte) {
	for _, value := range values {
		if o.iso15434Active && o.iso15434FormatCount < len(o.iso15434Format) {
			o.iso15434Format[o.iso15434FormatCount] = value
			o.iso15434FormatCount++
		}
		if o.dataCount < len(o.leading) {
			o.leading[o.dataCount] = value
		}
		o.dataCount++
		o.raw = append(o.raw, value)
		if o.variant.UsesISO23634Base() && value == '\\' {
			o.transmission = append(o.transmission, '\\')
		}
		o.transmission = append(o.transmission, value)
	}
}

func (o *messageOutput) appendECI(assignment int) {
	o.controls = append(o.controls, MessageControl{
		Kind: MessageControlECI, Offset: len(o.raw), Assignment: assignment,
	})
	o.transmission = append(o.transmission, '\\')
	var digits [6]byte
	for i := len(digits) - 1; i >= 0; i-- {
		digits[i] = byte('0' + assignment%10)
		assignment /= 10
	}
	o.transmission = append(o.transmission, digits[:]...)
}

func (o *messageOutput) fnc1() bool {
	if o.iso15434Used {
		return false
	}
	if o.fnc1Active {
		o.controls = append(o.controls, MessageControl{
			Kind: MessageControlFNC1Separator, Offset: len(o.raw),
		})
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
	o.controls = append(o.controls, MessageControl{
		Kind: MessageControlFNC1Start, Offset: len(o.raw),
	})
	return true
}

func (o *messageOutput) iso15434() bool {
	if o.iso15434Used || o.fnc1Modifier != 0 || o.dataCount != 0 {
		return false
	}
	o.iso15434Active = true
	o.iso15434Used = true
	o.controls = append(o.controls, MessageControl{
		Kind: MessageControlISO15434Start, Offset: len(o.raw),
	})
	// Table 15 represents the ISO/IEC 15434 message header with the switch.
	// Append it directly so the following two data characters remain the
	// format indicator tracked by appendData.
	o.transmission = append(o.transmission, '[', ')', '>', 30)
	return true
}

func (o *messageOutput) eot() bool {
	if o.fnc1Active {
		o.fnc1Active = false
		o.controls = append(o.controls, MessageControl{
			Kind: MessageControlFNC1End, Offset: len(o.raw),
		})
		return true
	}
	if !o.iso15434Active || o.iso15434FormatCount != len(o.iso15434Format) ||
		!isASCIIDigit(o.iso15434Format[0]) || !isASCIIDigit(o.iso15434Format[1]) {
		return false
	}
	o.iso15434Active = false
	if o.iso15434Format != [2]byte{'0', '2'} && o.iso15434Format != [2]byte{'0', '8'} {
		o.transmission = append(o.transmission, 4)
	}
	o.controls = append(o.controls, MessageControl{
		Kind: MessageControlISO15434End, Offset: len(o.raw),
	})
	return true
}

func (o *messageOutput) finish() (Message, bool) {
	if !o.variant.UsesISO23634Base() {
		return Message{
			Data: o.raw, ReaderTransmission: o.transmission,
			Controls: o.controls, Variant: o.variant,
		}, true
	}
	if o.fnc1Active || o.iso15434Active {
		return Message{}, false
	}
	modifier := o.fnc1Modifier
	if modifier == 0 {
		modifier = 1
	}
	data := make([]byte, 0, len(o.transmission)+3)
	data = append(data, ']', 'j', '0'+modifier)
	data = append(data, o.transmission...)
	return Message{
		Data: o.raw, ReaderTransmission: data,
		Controls: o.controls, Variant: o.variant,
	}, true
}

func (o *messageOutput) fail() (Message, bool) {
	if o.variant.UsesISO23634Base() {
		return Message{}, false
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
	data, _ := DecodeDataVariant(bits, wire.ISO23634)
	return data
}

// DecodeDataVariant interprets a corrected bit stream under the selected wire
// variant. ok is false when an ISO stream is truncated, uses a reserved switch,
// or violates an ISO/IEC 15434 or FNC1 start/end protocol. C-reference mode
// preserves the reference decoder's partial-message behavior.
func DecodeDataVariant(bits []byte, variant wire.Variant) ([]byte, bool) {
	message, ok := DecodeMessageVariant(bits, variant)
	return message.ReaderTransmission, ok
}

// DecodeMessageVariant interprets a corrected bit stream and returns both raw
// data and reader transmission without reparsing one representation from the
// other.
func DecodeMessageVariant(bits []byte, variant wire.Variant) (Message, bool) {
	// Ports decodeData in decoder.c.
	output := messageOutput{variant: variant}
	mode := spec.ModeUpper
	preMode := modeNone
	index := 0

	for index < len(bits) {
		if mode == modeECI || mode == modeFNC1 || mode == modeNone {
			// The C reference enters sentinel ECI/FNC1 modes that it does not
			// interpret. The ISO path handles those controls inline and cannot
			// normally reach these values. None is always an error state.
			if variant.UsesISO23634Base() {
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
				if variant.UsesISO23634Base() {
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
						if variant.UsesISO23634Base() {
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
						if !variant.UsesISO23634Base() {
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
						if !variant.UsesISO23634Base() {
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
							if !output.iso15434() {
								return output.fail()
							}
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
						if variant.UsesISO23634Base() {
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
						if variant.UsesISO23634Base() {
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
						if variant.UsesISO23634Base() {
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
					if variant.UsesISO23634Base() {
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
