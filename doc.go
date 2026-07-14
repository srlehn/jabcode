// Package jabcode is a pure-Go port of the JAB Code (Just Another Bar Code)
// reference library, a high-capacity 2D color matrix symbology standardized as
// ISO/IEC 23634:2022.
//
// The default encoder targets the experimental ISO/IEC 23634 wire profile. An
// untagged decoder accepts that profile; optional build tags add high-color,
// BSI, and historical C-reference decoder families without replacing ISO.
// Decode automatically tries every compiled family, while DecodeWithProfile
// forces one format.
package jabcode
