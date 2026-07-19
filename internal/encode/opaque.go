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
