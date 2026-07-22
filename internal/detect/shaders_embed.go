package detect

import _ "embed"

//go:embed shaders/halve_nrgba.wgsl
var halveNRGBAWGSL string

//go:embed shaders/histogram_rgb.wgsl
var histogramRGBWGSL string

//go:embed shaders/histogram_bounds.wgsl
var histogramBoundsWGSL string

//go:embed shaders/balance_rgb.wgsl
var balanceRGBWGSL string

//go:embed shaders/block_thresholds.wgsl
var blockThresholdsWGSL string

//go:embed shaders/binarize_rgb.wgsl
var binarizeRGBWGSL string

//go:embed shaders/filter_binary.wgsl
var filterBinaryWGSL string

//go:embed shaders/pack_binary_masks.wgsl
var packBinaryMasksWGSL string

//go:embed shaders/rotate_nrgba.wgsl
var rotateNRGBAWGSL string
