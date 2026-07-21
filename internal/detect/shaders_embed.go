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
