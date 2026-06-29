// Package jabcode is a pure-Go port of the JAB Code (Just Another Bar Code)
// reference library, a high-capacity 2D color matrix symbology standardized as
// ISO/IEC 23634:2022.
//
// The port mirrors the C reference implementation
// (https://github.com/jabcode/jabcode) closely enough to be bitstream- and
// image-compatible: codes produced here decode with the reference reader and
// vice versa.
package jabcode
