//go:build !jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/core"

const (
	currentCReadEnabled = false
	preV2CReadEnabled   = false
)

func decodePreV2CSampled(*core.Bitmap, *core.Bitmap, core.DecodedSymbol, *DiagnosticAttempt,
	func() ([3]*core.Bitmap, bool)) (*Message, bool) {
	return nil, false
}

func observePreV2CStreamSampled(*core.Bitmap, core.DecodedSymbol) ([]core.DecodedSymbol, primaryCorrection, bool, bool) {
	return nil, nil, false, false
}
