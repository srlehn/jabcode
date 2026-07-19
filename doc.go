// Package jabcode is a pure-Go port of the JAB Code (Just Another Bar Code)
// reference library, a high-capacity 2D color matrix symbology standardized as
// ISO/IEC 23634:2022.
//
// The default encoder targets the experimental ISO/IEC 23634 wire format. The
// dependency-light encoder subpackage provides the authoritative public write
// path for applications that do not need the reader; this root package keeps a
// facade over it. An untagged decoder accepts the ISO variant; optional build
// tags add high-color, BSI, and historical C-reference decoder capabilities
// without replacing ISO. Decode automatically uses every compiled capability.
// Forced single-variant decoding remains internal for CLI oracle and test work.
// The jabcode_non_iso_encode tag adds public ISO high-color and BSI encoder
// profiles without changing the untagged ISO default.
package jabcode
