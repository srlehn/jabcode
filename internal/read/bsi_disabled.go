//go:build !jabcode_bsi

package read

import "github.com/srlehn/jabcode/internal/core"

const bsiReadEnabled = false

func decodeBSISampled(*core.Bitmap, *core.Bitmap, core.DecodedSymbol, *DiagnosticAttempt) ([]byte, bool) {
	return nil, false
}
