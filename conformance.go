package jabcode

import "github.com/srlehn/jabcode/internal/wire"

// ConformanceMode selects behavior where the reference C implementation and
// ISO/IEC 23634:2022 differ.
type ConformanceMode uint8

const (
	// ConformanceCReference preserves compatibility with the reference C tools
	// and jabcode.org. It is the default.
	ConformanceCReference ConformanceMode = iota
	// ConformanceISO23634 selects ISO/IEC 23634:2022 wire behavior, including
	// its ECI and FNC1 decode and transmitted-data protocols.
	ConformanceISO23634
)

func (m ConformanceMode) profile() wire.Profile { return wire.Profile(m) }

func (m ConformanceMode) valid() bool { return m.profile().Valid() }
