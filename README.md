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

Single- and multi-symbol encode/decode work, including 4- and 8-color
portable symbols, docked secondary symbols, diagnostics, and a camera-stream
decoder. The main active work is print-capture robustness and future ISO-vs-C
conformance mode support.

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

For camera preview streams, use `jabcode.NewStream()`. It reuses the previous
frame's successful read hypothesis before falling back to a full search.

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

- 4- and 8-color symbols are the portable modes and are intended to round-trip
  with the reference C tools.
- 16- through 256-color symbols are a Go-only digital extension. They are useful
  only for pixel-exact images and are not expected to survive camera capture,
  print, or lossy compression.
- The default wire-format contract follows the C reference where it differs
  from ISO/IEC 23634. `ConformanceISO23634` and CLI `--conformance iso` expose
  an experimental ISO-target profile; it is not yet independently verified as
  strict conformance.
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
