package encode

import (
	"errors"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

const maxByteRunLength = 8207

// OpaqueCapacity returns the exact number of bytes that the fixed primary
// symbol in cfg can carry when every input byte is encoded in byte mode. The
// caller must provide a valid ISO-family, single-symbol configuration with an
// explicit version and resolved ECC level.
func OpaqueCapacity(cfg Config) (int, error) {
	if cfg.Format != wire.EncodeISO23634 && cfg.Format != wire.EncodeISOHighColor {
		return 0, errors.New("jabcode: opaque plans require an ISO-family encoding format")
	}
	if cfg.SymbolNumber != 1 || len(cfg.SymbolVersions) != 1 {
		return 0, errors.New("jabcode: opaque plans require one explicit primary symbol")
	}
	version := cfg.SymbolVersions[0]
	if version.X < 1 || version.X > 32 || version.Y < 1 || version.Y > 32 {
		return 0, errors.New("jabcode: opaque plan has an invalid primary version")
	}
	if cfg.ECCLevel < 1 || cfg.ECCLevel >= len(spec.ECCWeights) {
		return 0, errors.New("jabcode: opaque plan needs a resolved ECC level")
	}

	e := encoder{
		colors:   cfg.Colors,
		eccLevel: cfg.ECCLevel,
		format:   cfg.Format,
		opaque:   true,
	}
	capacity := e.symbolCapacity(version, true)
	weights := spec.ECCWeights[cfg.ECCLevel]
	messageBits := netCapacity(capacity, weights[0], weights[1]) - 5
	if messageBits < opaqueBitLength(1) {
		return 0, errors.New("jabcode: fixed symbol has no opaque-byte capacity")
	}

	low, high := 1, messageBits/8
	for low < high {
		mid := low + (high-low+1)/2
		if opaqueBitLength(mid) <= messageBits {
			low = mid
		} else {
			high = mid - 1
		}
	}
	return low, nil
}

func opaqueBitLength(length int) int {
	bits := length * 8
	for length > 0 {
		run := min(length, maxByteRunLength)
		bits += 7 + 4
		if run > 15 {
			bits += 13
		}
		length -= run
	}
	return bits
}

// encodeOpaqueData emits one or more upper-to-byte shifts. Each run has a
// content-independent header, so opaqueBitLength is exact for every byte
// slice of the same length.
func encodeOpaqueData(data []byte) []byte {
	encoded := make([]byte, opaqueBitLength(len(data)))
	position := 0
	for len(data) > 0 {
		runLength := min(len(data), maxByteRunLength)
		writeBits(encoded, 124, position, 7)
		position += 7
		writeBits(encoded, cap15(runLength), position, 4)
		position += 4
		if runLength > 15 {
			writeBits(encoded, runLength-16, position, 13)
			position += 13
		}
		for _, value := range data[:runLength] {
			writeBits(encoded, int(value), position, 8)
			position += 8
		}
		data = data[runLength:]
	}
	return encoded
}

func structuredControlBits(control Control) (int, error) {
	switch control.Kind {
	case ControlECI:
		if control.Assignment < 0 || control.Assignment > 999999 {
			return 0, errors.New("jabcode: ECI assignment must be between 0 and 999999")
		}
		switch {
		case control.Assignment <= 127:
			return 15, nil
		case control.Assignment <= 16383:
			return 23, nil
		default:
			return 29, nil
		}
	case ControlFNC1Start, ControlFNC1Separator, ControlFNC1End:
		return 10, nil
	default:
		return 0, errors.New("jabcode: unknown structured encoder control")
	}
}

func writeStructuredControl(bits []byte, position *int, control Control) {
	switch control.Kind {
	case ControlECI:
		writeBits(bits, 126, *position, 7)
		*position += 7
		switch {
		case control.Assignment <= 127:
			writeBits(bits, control.Assignment, *position, 8)
			*position += 8
		case control.Assignment <= 16383:
			writeBits(bits, (2<<14)|control.Assignment, *position, 16)
			*position += 16
		default:
			writeBits(bits, (3<<20)|control.Assignment, *position, 22)
			*position += 22
		}
	case ControlFNC1Start, ControlFNC1Separator, ControlFNC1End:
		writeBits(bits, 31, *position, 5)
		*position += 5
		writeBits(bits, 3, *position, 2)
		*position += 2
		additional := 4
		if control.Kind == ControlFNC1End {
			additional = 5
		}
		writeBits(bits, additional, *position, 3)
		*position += 3
	}
}

// encodeStructuredData keeps the existing mode optimizer out of the control
// grammar. Each data segment uses byte mode and returns to upper mode, making
// control placement independent of the preceding segment's optimizer state.
func encodeStructuredData(data []byte, controls []Control) ([]byte, error) {
	position := 0
	cursor := 0
	active := false
	dataCount := 0
	var leading [2]byte
	total := 0
	for _, control := range controls {
		if control.Offset < cursor || control.Offset > len(data) {
			return nil, errors.New("jabcode: structured control offset is out of order")
		}
		bits, err := structuredControlBits(control)
		if err != nil {
			return nil, err
		}
		for _, value := range data[cursor:control.Offset] {
			if dataCount < len(leading) {
				leading[dataCount] = value
			}
			dataCount++
		}
		total += opaqueBitLength(control.Offset-cursor) + bits
		switch control.Kind {
		case ControlECI:
		case ControlFNC1Start:
			if active || !validFNC1Start(dataCount, leading) {
				return nil, errors.New("jabcode: invalid FNC1 start placement")
			}
			active = true
		case ControlFNC1Separator:
			if !active {
				return nil, errors.New("jabcode: FNC1 separator without an active FNC1 message")
			}
			dataCount++
		case ControlFNC1End:
			if !active {
				return nil, errors.New("jabcode: FNC1 end without an active FNC1 message")
			}
			active = false
		}
		cursor = control.Offset
	}
	if active {
		return nil, errors.New("jabcode: unterminated FNC1 message")
	}
	total += opaqueBitLength(len(data) - cursor)
	encoded := make([]byte, total)
	cursor = 0
	for _, control := range controls {
		segment := data[cursor:control.Offset]
		copyBits(encoded, &position, encodeOpaqueData(segment))
		writeStructuredControl(encoded, &position, control)
		cursor = control.Offset
	}
	copyBits(encoded, &position, encodeOpaqueData(data[cursor:]))
	return encoded, nil
}

func copyBits(dst []byte, position *int, src []byte) {
	for _, bit := range src {
		dst[*position] = bit
		(*position)++
	}
}

func validFNC1Start(dataCount int, leading [2]byte) bool {
	switch dataCount {
	case 0:
		return true
	case 1:
		return (leading[0] >= 'A' && leading[0] <= 'Z') || (leading[0] >= 'a' && leading[0] <= 'z')
	case 2:
		return leading[0] >= '0' && leading[0] <= '9' && leading[1] >= '0' && leading[1] <= '9'
	default:
		return false
	}
}
