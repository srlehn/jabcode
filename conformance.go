package jabcode

import "github.com/srlehn/jabcode/internal/wire"

// ConformanceMode selects a wire-format behavior profile where the reference C
// implementation and ISO/IEC 23634:2022 differ.
type ConformanceMode uint8

const (
	// ConformanceCReference preserves compatibility with the reference C tools
	// and jabcode.org. It is the default.
	ConformanceCReference ConformanceMode = iota
	// ConformanceISO23634 selects the experimental profile targeting ISO/IEC
	// 23634:2022, including its ECI and FNC1 reader-transmission protocols. It
	// is not yet verified strict conformance: ISO/IEC 15434 message framing is
	// not implemented and the Annex F range reduction lacks an independent
	// wire oracle.
	ConformanceISO23634
)

func (m ConformanceMode) profile() wire.Profile { return wire.Profile(m) }

func (m ConformanceMode) valid() bool { return m.profile().Valid() }
