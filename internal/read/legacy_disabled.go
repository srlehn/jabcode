//go:build !jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/core"

const legacyReadEnabled = false

func decodeLegacyBitmap(_ *core.Bitmap, _ [3]*core.Bitmap, _ func() bool, _ *finding, _ *DiagnosticAttempt) ([]byte, readStage, bool) {
	return nil, readNoFinders, false
}
