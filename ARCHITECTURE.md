# Architecture

This document gives a high-level map of the codebase for people who want to
modify it. It describes how the pieces fit together and the invariants that hold
across them; it deliberately avoids line numbers and most function names, which
churn. For the *what* and *why* of any single routine, read the code - every
function ported from the reference C library names its counterpart in a
`// Ports ...` comment at the top of its body.

## Overview

`github.com/srlehn/jabcode` is a pure-Go port of [JAB Code][jabcode] (Just
Another Bar Code), the polychrome 2-D matrix symbology standardised as
ISO/IEC 23634:2022. It encodes bytes into a colour matrix image and decodes such
images back into bytes.

The port's contract is **behavioural compatibility with the reference C
library** ([github.com/jabcode/jabcode][jabcode]), not with the prose of the ISO
standard. Where the C library diverges from the standard, the port follows the C
library, so that codes round-trip with the existing JAB ecosystem. The known
divergences are listed under [Invariants](#invariants-and-cross-cutting-concerns).
On the decode side the port additionally goes **beyond** the C reference in
robustness - it reads rotated, screen-photographed and colour-cast captures the
C reader does not - without changing the wire format (see
[Robustness extensions](#robustness-extensions-beyond-the-c-reference)).

The code is a small public package over a set of internal packages, plus thin
command-line front ends. The public API is deliberately small:

- `Encoder`, built with `NewEncoder(...Option)`, and its `Encode([]byte)`
  method (bytes to `image.Image`). Options: `WithColors`, `WithECCLevel`,
  `WithModuleSize`, `WithSymbols`.
- `Decode(image.Image)` - image back to bytes.

Everything else lives under `internal/`.

## Package layout

- **root (`jabcode`)** - public API: `Decode` (decode.go) and `Encoder`
  (encoder.go) plus input validation; thin wrappers over the internal packages.
- **`internal/encode`** - the whole write path: data analysis/encoding, module
  placement, masking, multi-symbol cascade, rendering.
- **`internal/decode`** - the whole read path: bitmap, binarization,
  finder/alignment detection, rotation recovery, sampling, metadata/palette
  decode, secondaries.
- **`internal/ecc`** - LDPC construction/encode/decode (hard and soft),
  interleaving, and the fixed-seed PRNG they share.
- **`internal/palette`** - the 4- and 8-colour palettes.
- **`internal/spec`** - symbol-layout constants and pure layout arithmetic
  (side sizes, metadata walk, mask values).
- **`internal/tables`** - the spec-derived constant tables (alignment
  positions, palette placement, colour-mode parameters, ...).
- **`internal/testutil`** - shared test-fixture access (central `testdata/`).
- **`cmd/jabcodeWriter`, `cmd/jabcodeReader`** - CLI wrappers over `Encoder` /
  `Decode`; **`cmd/jabdecode`** - minimal decode CLI.
- **`internal/cmd/jabdiag`** - detector diagnostic: runs `decode.Diagnose` on
  the capture named by `JABDIAG_IMG`, dumping per-stage detection/decode
  evidence.

## Bird's-eye view

There are exactly two flows. Read them top to bottom; the file names point at
where each step lives.

### Encode (bytes -> image)

```text
NewEncoder + Options           encoder.go (root)
        |
        v
  analyse + encode data        internal/encode/encode_data.go
        |                      (mode selection, bit stream)
        v
  LDPC error correction        internal/ecc/ldpc.go
        |
        v
  interleave + pad             internal/ecc/interleave.go, random.go
        |
        v
  place modules                internal/encode/matrix.go, nondefault.go
  (data, palettes, metadata)   internal/palette, internal/spec, internal/tables
        |
        v
  data masking                 internal/encode/mask.go
        |
        v
  render bitmap -> image       internal/encode/encode.go
```

For multi-symbol codes (a primary plus docked secondaries) the same steps run
per symbol, orchestrated by `internal/encode/encoder_multi.go`.

### Decode (image -> bytes)

The entry point is coarse-to-fine: a full-resolution upright read first (clean
captures resolve here and stay byte-identical with the C reference), then - only
on failure - a cheap orientation search on a downscaled copy, then full
resolution again on the few promising orientations.

```text
Decode(img)                    internal/decode/detectprimary.go
        |
        v
  upright decodeImage          one full read in one coherent image frame
        |         \
        | fail     \ success -> bytes
        v
  finder-evidence bailout      blank/uniform images skip the rotation search
        |
        v
  coarse orientation probe     internal/decode/coarse.go, rotate.go
  (512px copy, 15-degree       cross-check survivors discriminate the angle;
  rungs over a 90-degree       each retained family expands to its four
  window)                      90-degree turns
        |
        v
  full-res decodeImage per rung until one reads
```

Inside one `decodeImage` pass:

```text
  binarize + classify colours  bitmap.go, binarizer.go
  (scale-adaptive per-channel  (plus descreen.go / pitch.go retries seeded by
  block-mean thresholds)       an autocorrelation pitch estimate)
        |
        v
  locate finder patterns       detector.go, finderpattern.go, findprimary.go
  (+ recover a missing one)    detector_recovery.go
        |
        v
  geometric quad consensus     finderquad.go
  (retry when per-type         (exhaustive type-correct 4-tuple search scored
  selection is incoherent)     by convexity, edge agreement, module size)
        |
        v
  locate alignment patterns    detector_ap.go
        |
        v
  perspective + sample grid    transform.go, sample.go
        |
        v
  read metadata + palettes     decsym.go, paldecode.go, internal/ecc/ldpc_soft.go
  (Part I falls back to        (finder-core colour references recover Part I
  finder-core references)      under a display colour cast)
        |
        v
  demask + deinterleave + LDPC decoder.go, internal/ecc
        |
        v
  decode modes -> message      decoder.go, internal/encode/encode_data.go
        |
        v
  recurse into docked          detector_secondary.go, decoder_secondary.go
  secondary symbols
```

As the last resort, the same orientation search runs per region of interest
(`roi.go`, joint chroma-variance x gradient-energy tile score): a symbol small
within a large frame vanishes in the whole-frame probe downscale, and probing
the proposed region at its own scale restores the module resolution the probe
needs.

## Code map

### Public surface (root package)

- **`encoder.go`** - the `Encoder` type, functional `Option`s, input validation.
- **`decode.go`** - the package-level `Decode` entry point.
- **`doc.go`** - package documentation.

### `internal/encode`

- **`encode_data.go`** - character analysis and the trellis that chooses the
  cheapest sequence of encoding modes, then emits the raw bit stream.
- **`encode.go`** - the single-symbol pipeline and bitmap -> image rendering,
  including `Render` (matrix + palette ground truth for tests/harness).
- **`matrix.go`** - module placement: finder/alignment patterns, the
  metadata/palette walk, and the data region.
- **`mask.go`** - the eight data-mask patterns and penalty scoring.
- **`nondefault.go`** - error-correction weight selection and metadata
  building/placement for non-default symbol parameters.
- **`encoder_multi.go`** - multi-symbol cascade: docking geometry, per-symbol
  metadata, data distribution, combined rendering.

### `internal/decode`

- **`bitmap.go`** - the raw RGB pixel buffer the detector works on.
- **`binarizer.go`** - white/black-point balancing and per-channel colour
  binarization against a scale-adaptive grid of interpolated block means.
- **`descreen.go`, `pitch.go`** - linear-light low-pass retries sized by a
  per-image autocorrelation estimate of the screen/subpixel lattice pitch.
- **`detector.go`** - the run-length state machines recognising finder-pattern
  cross-sections in a scanline.
- **`finderpattern.go`** - finder-pattern geometry, cross-checks, typing.
- **`findprimary.go`** - assembling four typed finder patterns into a primary
  symbol; interpolating a single missing one.
- **`detector_recovery.go`** - local search for a missing finder pattern.
- **`finderquad.go`** - geometric finder-quad consensus retry.
- **`detector_ap.go`** - alignment-pattern detection and resampling.
- **`detector_secondary.go`** - geometry of docked secondary symbols.
- **`coarse.go`, `rotate.go`** - the downscaled orientation probe and the
  rotation primitive behind the coarse-to-fine `Decode`.
- **`roi.go`** - region-of-interest proposals: the tile scoring behind
  `Decode`'s last-resort per-region retry and the ROI diagnostics.
- **`transform.go`** - the perspective transform between image and module space.
- **`sample.go`** - sampling module colours on the established grid.
- **`decsym.go`** - symbol metadata decode (Part I/II) and the data-map of
  reserved modules; Part I retries classification against finder-core colour
  references when absolute thresholds fail under a colour cast.
- **`paldecode.go`** - embedded-palette reading, palette-referenced module
  classification, and the finder-core reference synthesis.
- **`decoder.go`** - sampled modules to bits: demask -> deinterleave -> LDPC ->
  mode decode -> message.
- **`decoder_secondary.go`** - secondary-symbol palette reading and decode.
- **`detectprimary.go`** - `Decode`, `decodeImage`, and the primary-symbol
  detection orchestration (`primaryDetector` with its observation-only stats).
- **`diag.go`** - `Diagnose`: the staged evidence dump behind
  `internal/cmd/jabdiag`; never influences decoding.

### `internal/ecc`

- **`ldpc.go`** - LDPC code construction (Gallager + Gauss-Jordan), encoding,
  hard-decision decoding. **`ldpc_soft.go`** - log-domain belief propagation for
  the metadata LDPC. **`bitmatrix.go`** - dense GF(2) matrix.
- **`interleave.go`**, **`random.go`** - fixed-seed PRNG-driven (de)interleaving.

### Commands and fixtures

- **`cmd/jabcodeWriter`**, **`cmd/jabcodeReader`**, **`cmd/jabdecode`** - CLIs.
- **`internal/cmd/jabdiag`** - detector diagnostic (`JABDIAG_IMG` names the
  capture; nothing in the tree hard-codes private photo paths).
- **`testdata/`** - golden vectors (bit streams, matrices, palettes) checked
  against the C reference, clean C-encoded fixtures, and the detection
  snapshot golden; consumed via `internal/testutil`.

## Invariants and cross-cutting concerns

These hold across the whole module; breaking one is an architectural change, not
a local one.

- **C-reference compatibility is the contract for the wire format.** Encoder
  output must be bit/format compatible with the reference C library so codes
  interoperate. Ported functions name their C counterpart in a `// Ports ...`
  comment.
  Where the C library diverges from ISO/IEC 23634:2022, the port matches the
  C library; the known divergences are listed below.
- **Naming: primary/secondary.** The reference C library calls the two symbol
  roles "master"/"slave"; this port uses **primary**/**secondary** throughout
  (types, functions, files). Comments bridge to the old C names where helpful.
- **Determinism via fixed PRNG seeds.** Interleaving and LDPC matrix
  construction use the standard's fixed seeds (data-stream LDPC, metadata
  LDPC, and interleaving each have their own). The PRNG lives in
  `internal/ecc/random.go`. These seeds are part of the wire format - do not
  change them.
- **Colour-mode scope.** Only 4- and 8-colour symbols are produced and
  consumed. Validation rejects other colour counts before any table is
  indexed, so malformed input returns an error rather than panicking.
- **Never panic on any input.** `Decode` must return an error, not panic, on
  arbitrary images and on hostile/degenerate geometry. The port fixes unsafe
  C patterns rather than mirroring them; a fuzz-style robustness test guards
  this. The guarantee is fail-safe, not resource-bounded: decoding allocates
  working buffers proportional to the input's pixel count (the bitmap and the
  rotation/descreen copies), so callers decoding untrusted images should bound
  the image dimensions first.
- **Coordinate and image conventions.** Module coordinates are `image.Point`;
  pixel work uses `image.Image`/`color`; detection geometry uses an internal
  float point type. The encoder returns a paletted image.

### Known divergences from ISO/IEC 23634:2022 (following the C library)

*4-colour palette order.* The standard (Tables 4/21) orders the four colours
black, cyan, magenta, yellow; the C library (and therefore this port) uses
black, magenta, yellow, cyan. Because the palette is embedded in the symbol
and read back during decode, the index sequence still round-trips; only the
physical colours of a 4-colour symbol differ from a strict-spec one.

*More than 8 colours.* The standard's Annex G (informative) sketches
16-256-colour modes, but the C library cannot actually handle them: its
palette-placement table is sized for at most 8 colours and it indexes out of
bounds beyond that (undefined behaviour). With no real ecosystem to
interoperate with, the port scopes itself to exactly the 4- and 8-colour
modes; validation rejects other colour counts with an error.

*ECI / FNC1.* Decoding of these channels is only partially implemented, the
same as in the C reference.

### Robustness extensions beyond the C reference

The decode path deliberately exceeds the C reader's robustness while keeping
its behaviour on clean input byte-identical (the extensions run only on
failure, or reduce to the C behaviour on clean pixels):

- **Scale-adaptive per-channel binarization** - block-mean threshold grid
  derived from image size, replacing the C's coarse fixed-scale average.
- **Descreen retries** - lattice-pitch-estimated low-pass passes for screen
  subpixel/moire damage.
- **Coarse-to-fine rotation recovery** - the C reader's finder detection
  collapses beyond roughly 20 degrees of in-plane rotation; `Decode` recovers
  orientation via the downscaled probe and reads clean codes through at least
  60 degrees.
- **Geometric finder-quad consensus** - a retry selecting the four finder
  candidates that actually form a symbol quad when per-type selection is
  incoherent.
- **Part I metadata recovery via finder-core references** - when absolute
  colour thresholds fail under a display cast, the Nc modules are re-classified
  against references synthesized from the symbol's own finder cores (offset
  from the black cores, per-channel gains from the cyan/yellow cores).

Every scale-dependent value in these extensions (descreen kernel, binarization
block size, probe resolution) is **estimated from the image**, never a fixed
pixel constant.

## Verification strategy

Correctness is pinned several ways:

- **Golden vectors** under `testdata/` capture intermediate and final outputs
  (bit streams, placed matrices, palettes, interleave/PRNG sequences) and are
  asserted by the unit tests.
- **A C oracle.** During development the reference C library is built and linked
  into small oracle programs that exercise the same inputs, giving bidirectional
  Go-C interop checks (encode in one, decode in the other, both directions).
  These harnesses live outside the published tree; the golden vectors under
  `testdata/` are their committed output, so the C contract stays asserted by
  `go test` alone, without the C toolchain.
- **A detection snapshot test** pins the pre-finder chain on a clean in-memory
  encode, so detector refactors prove Go self-invariance.
- **A degradation regression harness** (`go test -tags jabharness`) encodes
  known payloads, applies seeded degradations (JPEG, blur, cast, noise,
  synthetic screen lattice, rotation), and buckets each run by the pipeline
  stage reached, plus a pre-LDPC module error rate against the encoder's
  ground-truth matrix.
- **The `jabdiag` diagnostic** replays every decode stage on a real capture
  with full evidence dumps, for measure-first debugging.

The test files alongside the sources cover round-trips (Go encode -> Go decode),
decoding C-produced vectors, multi-symbol cascades, soft- and hard-decision LDPC,
rotation recovery, and option validation.

## References

- ISO/IEC 23634:2022 - *Information technology - Automatic identification and
  data capture techniques - JAB Code polychrome bar code symbology specification*.
- Reference implementation: [github.com/jabcode/jabcode][jabcode].

[jabcode]: https://github.com/jabcode/jabcode
