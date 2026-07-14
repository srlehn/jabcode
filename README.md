# jabcode

[![PkgGoDev](https://pkg.go.dev/badge/github.com/srlehn/jabcode)](https://pkg.go.dev/github.com/srlehn/jabcode)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-srlehn%2Fjabcode-blue.svg)](https://deepwiki.com/srlehn/jabcode)
![experimental](https://img.shields.io/badge/status-experimental-orange.svg)

Pure-Go JAB Code encoder and decoder.

JAB Code is a high-capacity color matrix barcode standardized as
ISO/IEC 23634:2022.

This module is **experimental**. The API may still change, and scanning
real-world captures is still being hardened. When you find any errors,
please report them as issues.

## Status

Single- and multi-symbol encode/decode work, including normative 4- and 8-color
ISO modes, docked secondary symbols, diagnostics, and a camera-stream decoder.
Tagged builds add high-color, BSI and historical decoder families. The main
active work is print-capture robustness, stream integration, performance, and
validation of the experimental ISO target.

## Install

```sh
go get github.com/srlehn/jabcode
```

Install the CLI:

```sh
go install github.com/srlehn/jabcode/cmd/jabcode@latest
```

## Library

```go
package main

import (
    "bytes"
    "image/png"
    "os"

    "github.com/srlehn/jabcode"
)

func main() {
    img, err := jabcode.NewEncoder(
        jabcode.WithColors(8),
        jabcode.WithModuleSize(12),
    ).Encode([]byte("hello"))
    if err != nil {
        panic(err)
    }

    var buf bytes.Buffer
    if err := png.Encode(&buf, img); err != nil {
        panic(err)
    }
    if err := os.WriteFile("hello.png", buf.Bytes(), 0o644); err != nil {
        panic(err)
    }
}
```

Decoding accepts any `image.Image`, so file format support is provided by the
decoders registered by the caller.

```go
f, err := os.Open("hello.png")
if err != nil {
    panic(err)
}
defer f.Close()

img, err := png.Decode(f)
if err != nil {
    panic(err)
}

data, err := jabcode.Decode(img)
if err != nil {
    panic(err)
}
_ = data
```

For camera preview streams, use `jabcode.NewStream()`. It reuses previous read
hypotheses and compatible evidence within a fixed per-frame work budget.

## Commands

Encode payload bytes from stdin to PNG:

```sh
printf hello | jabcode encode --output hello.png
```

For shell demos, literal text input is also available:

```sh
jabcode encode --input "hello" --output hello.png
```

Decode an image to stdout:

```sh
jabcode decode hello.png
```

Write decoded bytes to a file:

```sh
jabcode decode --output payload.bin hello.png
```

`jabcode decode` registers PNG, JPEG, HEIC, AVIF, TIFF, and WebP decoders
(including WebP VP8 and VP8L).

Detector diagnostics for difficult captures write the payload to stdout and the
diagnostic report to stderr; annotated diagnostic images go to `--diag-out`.
The diagnostic mode observes the authoritative read once and does not replay a
second decode pipeline:

```sh
jabcode decode --diag --diag-out ./diag-images capture.png > payload.bin
```

Multi-symbol encodes use one compact symbol spec per symbol:

```sh
jabcode encode --symbols 0:4x4:0,2:4x4:0 --output cascade.png < payload.bin
```

## Compatibility

- The default encoder targets ISO/IEC 23634:2022 with the normative 4- and
  8-color modes. The ISO target remains experimental until the Annex F range
  reduction has an independent wire oracle.
- Decoder build tags are additive. Untagged `Decode` accepts ISO only;
  `jabcode_high_color`, `jabcode_bsi`, and `jabcode_legacy` add their compiled
  routes to the same automatic read. The CLI-only `--only` flag forces one
  compiled wire variant for oracle or debugging work.
- `jabcode_high_color` adds decoding of the non-standard ISO-derived 16-
  through 256-color modes. `jabcode_non_iso_encode` adds the public encoder
  profile selector with `hc` and `bsi` output. Use the corresponding decoder
  tag as well when the same binary must read what it writes. Physical
  robustness decreases with color density: measured capture
  limits range from camera-grade 16/32 colors to scanner-grade 128 colors,
  while 256 colors remain pixel-exact only. See `WithColors` for details.
- `jabcode_legacy` adds read-only current and pre-v2.0 C-reference formats,
  including docked multi-symbol codes. No legacy encoder is exposed.
- `jabcode_bsi` adds exact BSI TR-03137 primary and recursively docked-secondary
  decoding. `jabcode_non_iso_encode` exposes `ProfileBSI` and CLI
  `--profile bsi` for single- and multi-symbol output. BSI supports its
  specified 4- through 256-color layouts; the CLI warns above 8 colors because
  capture robustness still falls as palette density rises.
- `Decode` is intended to return errors, not panic, on malformed or hostile
  images. Callers should still bound untrusted image dimensions before decoding.

## Layout

- Root package: public `Encoder`, `Decode`, and `Stream`.
- `internal/encode`: data encoding, matrix placement, masking, and rendering.
- `internal/core`: shared pixel buffers, geometry, decoded-symbol types, and
  status values used by the read path.
- `internal/read`, `internal/detect`, `internal/decode`: image search,
  detection, sampling, metadata, palette, ECC, and payload decoding.
- `internal/diag`: staged text and image diagnostics over the decoder.
- `internal/ecc`, `internal/palette`, `internal/spec`, `internal/tables`:
  shared format machinery.
- `cmd/`: user-facing CLIs.

## Development

More detail:

- [ARCHITECTURE.md](ARCHITECTURE.md) describes the package boundaries,
  invariants, robustness extensions, and verification strategy.
- [WIRE_FORMAT.md](WIRE_FORMAT.md) records the C-reference wire format and
  known ISO and pre-ISO deltas.
