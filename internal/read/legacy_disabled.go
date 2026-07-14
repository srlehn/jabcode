//go:build !jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/core"

const legacyReadEnabled = false

func decodeLegacySampled(_ *core.Bitmap, _ [3]*core.Bitmap, _ *core.Bitmap, _ core.DecodedSymbol, _ *DiagnosticAttempt) ([]byte, bool) {
	return nil, false
}
