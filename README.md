# jabcode

[![PkgGoDev](https://pkg.go.dev/badge/github.com/srlehn/jabcode)](https://pkg.go.dev/github.com/srlehn/jabcode)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-srlehn%2Fjabcode-blue.svg)](https://deepwiki.com/srlehn/jabcode)
![experimental](https://img.shields.io/badge/status-experimental-orange.svg)

Pure-Go JAB Code encoder and decoder.

JAB Code is a high-capacity color matrix barcode standardized as
ISO/IEC 23634:2022.

This module is **experimental**. The API may still change, and scanning
real-world captures is still being hardened. When you find any errors please report them as issues.

## Status

Single- and multi-symbol encode/decode work, including 4- and 8-color
portable symbols, docked secondary symbols, diagnostics, and a camera-stream
decoder. The main active work is print-capture robustness and future ISO-vs-C
conformance mode support.

## Install

```sh
go get github.com/srlehn/jabcode
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

Encode text to PNG:

```sh
go run ./cmd/jabcodeWriter --input "hello" --output hello.png
```

Decode an image to stdout:

```sh
go run ./cmd/jabdecode hello.png
```

`jabdecode` registers PNG, JPEG, and HEIC decoders. `jabcodeReader` is a second
decode CLI with optional file output:

```sh
go run ./cmd/jabcodeReader hello.png --output payload.bin
```

Detector diagnostics for difficult captures:

```sh
JABDIAG_IMG=capture.png JABDIAG_OUT=./diag-images go run ./internal/cmd/jabdiag
```

## Compatibility

- 4- and 8-color symbols are the portable modes and are intended to round-trip
  with the reference C tools.
- 16- through 256-color symbols are a Go-only digital extension. They are useful
  only for pixel-exact images and are not expected to survive camera capture,
  print, or lossy compression.
- The current wire-format contract follows the C reference where it differs
  from ISO/IEC 23634. A strict ISO mode is planned but not implemented.
- `Decode` is intended to return errors, not panic, on malformed or hostile
  images. Callers should still bound untrusted image dimensions before decoding.

## Layout

- Root package: public `Encoder`, `Decode`, and `Stream`.
- `internal/encode`: data encoding, matrix placement, masking, and rendering.
- `internal/read`, `internal/detect`, `internal/decode`: image search,
  detection, sampling, metadata, palette, ECC, and payload decoding.
- `internal/ecc`, `internal/palette`, `internal/spec`, `internal/tables`:
  shared format machinery.
- `cmd/`: user-facing CLIs.

## Development

```sh
go test ./...
```

More detail:

- [ARCHITECTURE.md](ARCHITECTURE.md) describes the package boundaries,
  invariants, robustness extensions, and verification strategy.
- [WIRE_FORMAT.md](WIRE_FORMAT.md) records the C-reference wire format and
  known ISO and pre-ISO deltas.
