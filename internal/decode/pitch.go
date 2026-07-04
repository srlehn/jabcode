package decode

// EstimatePitch estimates the dominant lattice pitch of bm in pixels along the x
// and y axes, by 1-D autocorrelation of evenly sampled scanlines (for px) and
// columns (for py). On a screen capture this recovers the display's subpixel /
// diode-grid period without any prior detection, so the descreen low-pass can size
// its kernel per image rather than from a fixed radius, which would be wrong at
// some capture distance or display resolution. A returned 0 on an axis means no
// periodic structure was found in the search range — the caller should treat that
// as "no descreen on that axis".
func EstimatePitch(bm *Bitmap) (px, py int) {
	minDim := min(bm.Width, bm.Height)
	if minDim < 4 {
		return 0, 0
	}
	// The lattice pitch is a small fraction of the image; cap the lag search at
	// minDim/8. This is a *search bound* (it bounds cost), not the estimate — the
	// answer is the autocorrelation peak found inside it.
	maxLag := max(2, minDim/8)
	return dominantLag(sampleRows(bm), maxLag), dominantLag(sampleCols(bm), maxLag)
}

// pitchSampleLines caps how many scanlines/columns the estimator averages over, to
// bound cost on large captures; the lines are spread evenly across the image.
const pitchSampleLines = 32

// sampleRows returns up to pitchSampleLines evenly spaced rows of bm as luma
// (mean of R,G,B) signals.
func sampleRows(bm *Bitmap) [][]float64 {
	w, h, bpp := bm.Width, bm.Height, bm.Channels
	n := min(pitchSampleLines, h)
	lines := make([][]float64, 0, n)
	for k := range n {
		y := k * h / n
		base := y * w * bpp
		row := make([]float64, w)
		for x := range w {
			o := base + x*bpp
			row[x] = float64(int(bm.Pix[o])+int(bm.Pix[o+1])+int(bm.Pix[o+2])) / 3
		}
		lines = append(lines, row)
	}
	return lines
}

// sampleCols returns up to pitchSampleLines evenly spaced columns of bm as luma
// signals.
func sampleCols(bm *Bitmap) [][]float64 {
	w, h, bpp := bm.Width, bm.Height, bm.Channels
	n := min(pitchSampleLines, w)
	lines := make([][]float64, 0, n)
	for k := range n {
		x := k * w / n
		col := make([]float64, h)
		for y := range h {
			o := (y*w + x) * bpp
			col[y] = float64(int(bm.Pix[o])+int(bm.Pix[o+1])+int(bm.Pix[o+2])) / 3
		}
		lines = append(lines, col)
	}
	return lines
}

// dominantLag returns the lag in [1, maxLag] of the strongest biased
// autocorrelation summed over the given lines (each line's own mean removed). Two
// choices make this robust for period detection:
//   - The biased estimator (every lag divided by the full line length, not the
//     overlap) tapers long lags, so the fundamental period wins over its harmonics
//     and noisy large-lag estimates can't spike.
//   - The central lobe is skipped: the autocorrelation descends from lag 0 through
//     a valley near half the period before rising to the first true peak, so the
//     search starts at that valley rather than at lag 1.
//
// Returns 0 when there is no periodic peak (a flat or aperiodic axis).
func dominantLag(lines [][]float64, maxLag int) int {
	if len(lines) == 0 || maxLag < 2 {
		return 0
	}
	acf := make([]float64, maxLag+1)
	for _, s := range lines {
		n := len(s)
		if n < 2 {
			continue
		}
		var mean float64
		for _, v := range s {
			mean += v
		}
		mean /= float64(n)
		inv := 1 / float64(n)
		hi := min(maxLag, n-1)
		for lag := 0; lag <= hi; lag++ {
			var sum float64
			for x := 0; x+lag < n; x++ {
				sum += (s[x] - mean) * (s[x+lag] - mean)
			}
			acf[lag] += sum * inv
		}
	}
	// Walk down the central lobe to the first valley (where the curve turns up).
	lag := 1
	for lag < maxLag && acf[lag] >= acf[lag+1] {
		lag++
	}
	if lag >= maxLag {
		return 0 // monotonic descent: no periodic peak
	}
	best, bestVal := 0, 0.0
	for ; lag <= maxLag; lag++ {
		if acf[lag] > bestVal {
			bestVal, best = acf[lag], lag
		}
	}
	return best
}
