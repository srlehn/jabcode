//go:build !jabcode_bsi

package read

import "github.com/srlehn/jabcode/internal/core"

const bsiReadEnabled = false

func decodeBSISampled(*core.Bitmap, *core.Bitmap, core.DecodedSymbol, *DiagnosticAttempt, func() ([3]*core.Bitmap, bool)) ([]byte, bool) {
	return nil, false
}

func observeBSIStreamSampled(*core.Bitmap, core.DecodedSymbol) ([]core.DecodedSymbol, primaryCorrection, bool, bool) {
	return nil, nil, false, false
}
