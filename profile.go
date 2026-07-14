package jabcode

import (
	"errors"
	"fmt"

	"github.com/srlehn/jabcode/internal/wire"
)

// Profile selects one complete JAB Code wire format. Build tags make optional
// profiles available but never change the zero value or default selection.
type Profile uint8

const (
	// ProfileISO23634 selects the experimental ISO/IEC 23634:2022 target. It is
	// the zero value and default. Annex F range reduction still lacks an
	// independent wire oracle.
	ProfileISO23634 Profile = iota
	// ProfileHighColor extends the ISO target through 256 module colors. It
	// requires the jabcode_high_color build tag.
	ProfileHighColor
	// ProfileBSI selects exact BSI TR-03137 behavior. It is unavailable until
	// that separately tested implementation is compiled with jabcode_bsi.
	ProfileBSI
	// ProfileLegacy selects read-only historical C-reference behavior. It
	// requires the jabcode_legacy build tag.
	ProfileLegacy
)

var (
	// ErrProfileUnavailable reports that an optional profile was not compiled
	// into the current build.
	ErrProfileUnavailable = errors.New("jabcode: profile unavailable")
	// ErrProfileReadOnly reports an attempt to encode with a decode-only
	// compatibility profile.
	ErrProfileReadOnly = errors.New("jabcode: profile is read-only")
)

func (p Profile) String() string {
	switch p {
	case ProfileISO23634:
		return "iso"
	case ProfileHighColor:
		return "high_color"
	case ProfileBSI:
		return "bsi"
	case ProfileLegacy:
		return "legacy"
	default:
		return fmt.Sprintf("profile(%d)", p)
	}
}

// Available reports whether the selected decoder profile was compiled into
// the current build.
func (p Profile) Available() bool {
	switch p {
	case ProfileISO23634:
		return true
	case ProfileHighColor:
		return highColorProfileAvailable
	case ProfileBSI:
		return bsiProfileAvailable
	case ProfileLegacy:
		return legacyProfileAvailable
	default:
		return false
	}
}

func (p Profile) profile() wire.Profile {
	switch p {
	case ProfileISO23634:
		return wire.ISO23634
	case ProfileHighColor:
		return wire.HighColor
	case ProfileBSI:
		return wire.BSI
	case ProfileLegacy:
		return wire.Legacy
	default:
		return wire.Profile(255)
	}
}

func (p Profile) valid() bool { return p.profile().Valid() }

func (p Profile) validateAvailable() error {
	if !p.valid() {
		return fmt.Errorf("jabcode: invalid profile %d", p)
	}
	if !p.Available() {
		return fmt.Errorf("%w: %s", ErrProfileUnavailable, p)
	}
	return nil
}

func (p Profile) validateEncode() error {
	if err := p.validateAvailable(); err != nil {
		return err
	}
	if p == ProfileLegacy {
		return fmt.Errorf("%w: %s", ErrProfileReadOnly, p)
	}
	return nil
}
