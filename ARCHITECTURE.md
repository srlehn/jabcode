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

The port's contract today is **behavioural compatibility with the reference C
library** ([github.com/jabcode/jabcode][jabcode]), not with the prose of the ISO
standard. Where the C library diverges from the standard, the port follows the C
library, so that codes round-trip with the existing JAB ecosystem. Planned: a
caller-selectable conformance mode - C-reference compatibility (likely the
default while JAB Code adoption is young) versus strict ISO/IEC 23634
adherence. The known divergences are listed under
[Invariants](#invariants-and-cross-cutting-concerns).
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
- `Stream`, built with `NewStream()` - `Decode` for successive camera frames
  of one scene: it replays the previous frame's winning read hypothesis
  (resolution level and orientation) before falling back to the full search.

Everything else lives under `internal/`.

## Package layout

- **root (`jabcode`)** - public API: `Decode` (decode.go), `Stream`
  (stream.go) and `Encoder` (encoder.go) plus input validation; thin wrappers
  over the internal packages.
- **`internal/encode`** - the whole write path: data analysis/encoding, module
  placement, masking, multi-symbol cascade, rendering.
- **`internal/core`** - the shared read-path types leaf: pixel `Bitmap`,
  floating-point geometry (`PointF`, `Perspective`), the decoded-symbol
  result types, status codes, per-pixel colour statistics. Imports none of
  the packages below.
- **`internal/detect`** - symbol location: channel balancing, binarization
  (with pitch-estimated descreen retries), finder/alignment detection,
  side-size estimation, perspective sampling, the coarse orientation probe,
  the region-of-interest proposer.
- **`internal/decode`** - sampled matrix to message bits: metadata and
  palette decode, module colour classification, demask/deinterleave/LDPC,
  mode decoding, for primary and secondary symbols.
- **`internal/read`** - the coordinator joining the two: orientation and
  region retries, the detect-then-decode handoff (including the
  alignment-pattern fallback that needs the decoded side version), the
  docked-secondary walk. The coupling between detect and decode is
  orchestration, so it lives here rather than as an import between them.
- **`internal/diag`** - the staged decoder diagnostic behind jabdiag; reads
  the exported hooks of detect/decode/read, influences nothing.
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
- **`internal/cmd/jabdiag`** - detector diagnostic: runs `diag.Diagnose` on
  the capture named by `JABDIAG_IMG`, dumping per-stage detection/decode
  evidence; with `JABDIAG_OUT` set, also numbered per-stage annotated
  images.

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

`Decode` searches a resolution pyramid: the frame is converted once and
box-halved into levels down to a shorter-side floor (small frames stay
single-level and behave exactly as a pipeline without a pyramid). One
goroutine per level runs the level's upright read and then, on failure with
finder evidence, the level's orientation and region search. The coarsest
level also publishes its detection finding - the finder quad, module side
size and rung angle, in level coordinates - and a seeded route resumes from
that geometry on the finer levels (`seeded.go`): scale the quad, rotate the
level, sample and decode, with no finder search at fine resolution. When
the coarse route decoded, the seeded route reports success only if its
re-decode agrees byte-for-byte - two scales reading the same bytes through
the LDPC syndrome gate is stronger evidence than either alone; when the
coarse route could only locate (small-module captures), the seeded decode
stands on its own. Results commit in a fixed priority order - the coarsest
upright, the seeded route, the finer uprights (coarsest first), then every
search (coarsest first), never first-done - so the outcome is deterministic
regardless of scheduling; the seeded route reads only the coarsest level's
deterministic finding, and slots that can no longer win are told to quit at
their next stage boundary. Uprights outrank the rest because they are the
cheap bounded hypothesis, and the seeded route outranks the blind ladders,
which is what frees a locatable capture from waiting on the expensive
full-resolution upright ladder. A large capture rarely needs its full
resolution, so the common case returns in a coarse level's time; each
halving is also a mild low-pass, so coarse levels can read captures whose
full-resolution noise defeats detection.

Within one level the search is coarse-to-fine: the upright read first (clean
captures resolve here and stay byte-identical with the C reference), then -
only on failure - a cheap orientation search on a downscaled copy, then the
level's resolution again on the few promising orientations.

```text
Decode(img)                    internal/read/read.go, pyramid.go
        |
        v
  resolution pyramid           box-halved levels, one goroutine per level;
  (small frames: one level)    commit order: coarsest upright, seeded route,
        |                      finer uprights, searches
        v  per level:
  upright read                 one full read in one coherent image frame
        |         \            (the coarsest level publishes its finder quad;
        | fail     \           the seeded route re-decodes from it on finer
        |           \          levels and commits on cross-scale agreement -
        |            \         seeded.go)
        |             \ success -> bytes (once higher-priority slots failed)
        v
  finder-evidence bailout      blank/uniform levels stop here
        |
        v
  coarse orientation probe     internal/detect/coarse.go, rotate.go
  (512px copy, 15-degree       cross-check survivors discriminate the angle;
  rungs over a 90-degree       each retained family expands to its four
  window)                      90-degree turns
        |
        v
  level-resolution read per rung until one reads
```

Inside one `DecodeImage` pass (detection in `internal/detect`, matrix decoding
in `internal/decode`, the handoff in `internal/read`):

```text
  binarize + classify colours  core/bitmap.go, detect/binarizer.go
  (scale-adaptive per-channel  (plus descreen.go / pitch.go retries seeded by
  block-mean thresholds)       an autocorrelation pitch estimate)
        |
        v
  locate finder patterns       detect/detector.go, finderpattern.go,
  (+ recover a missing one)    findprimary.go, detector_recovery.go
        |
        v
  geometric quad consensus     detect/finderquad.go
  (retry when per-type         (exhaustive type-correct 4-tuple search scored
  selection is incoherent)     by convexity, edge agreement, module size)
        |
        v
  locate alignment patterns    detect/detector_ap.go
        |
        v
  perspective + sample grid    core/transform.go, detect/sample.go
        |
        v
  read metadata + palettes     decode/decsym.go, paldecode.go,
  (Part I falls back to        internal/ecc/ldpc_soft.go (finder-core colour
  finder-core references)      references recover Part I under a colour cast)
        |
        v
  demask + deinterleave + LDPC decode/decoder.go, internal/ecc
        |
        v
  decode modes -> message      decode/decoder.go, internal/encode/encode_data.go
        |
        v
  recurse into docked          read/read.go, detect/detector_secondary.go,
  secondary symbols            decode/decoder_secondary.go
```

As the last resort, the same orientation search runs per region of interest
(`detect/roi.go`, joint chroma-variance x gradient-energy tile score): a symbol
small within a large frame vanishes in the whole-frame probe downscale, and
probing the proposed region at its own scale restores the module resolution the
probe needs.

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

### `internal/core`

- **`bitmap.go`** - the raw RGB pixel buffer the read path works on.
- **`transform.go`** - `PointF` and the perspective transform between image
  and module space.
- **`symbol.go`** - `Metadata`, `DecodedSymbol` and the shared status codes.
- **`colorstats.go`** - per-pixel colour statistics shared by binarization
  and palette classification.

### `internal/detect`

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
  `Decode`'s last-resort per-region retry.
- **`sample.go`** - sampling module colours on the established grid.
- **`detectprimary.go`** - `PrimaryDetector` with its observation-only stats
  and the binarization-retry ladder (`LocateFinders`).
- **`sidesize.go`** - side-size estimation from finder-pair distances.

### `internal/decode`

- **`decsym.go`** - symbol metadata decode (Part I/II) and the data-map of
  reserved modules; Part I retries classification against finder-core colour
  references when absolute thresholds fail under a colour cast.
- **`paldecode.go`** - embedded-palette reading, palette-referenced module
  classification, and the finder-core reference synthesis.
- **`decoder.go`** - sampled modules to bits: demask -> deinterleave -> LDPC ->
  mode decode -> message.
- **`decoder_secondary.go`** - secondary-symbol palette reading and decode.

### `internal/read`

- **`read.go`** - `Decode` and `DecodeImage`: the orientation and
  region-of-interest retries, the detect-then-decode primary handoff with the
  alignment-pattern fallback, and the docked-secondary walk.
- **`pyramid.go`** - the resolution pyramid: level construction (one base
  conversion, box-halved levels above a shorter-side floor) and the
  concurrent per-level search with ordered commit (coarsest upright, seeded
  route, finer uprights, searches) and stage-boundary cancellation.
- **`seeded.go`** - the seeded route: re-enter the decode at the coarsest
  level's published finder quad on a finer level (scale, rotate, sample,
  decode - no fine finder search), committing on cross-scale byte agreement
  or, for a locate-only finding, on its own decode.
- **`stream.go`** - `Stream`: frame-sequence decoding that replays the
  previous frame's winning hypothesis (level shorter side plus pre-rotation)
  as one cheap decode before the full pyramid search; the prior survives a
  failed frame. Deterministic per frame given the frames before it.

### `internal/diag`

- **`diag.go`** - `Diagnose`: the staged evidence dump behind
  `internal/cmd/jabdiag`; never influences decoding.
- **`diagimg.go`** - the per-stage annotated image sink behind `Diagnose`'s
  image-directory mode (region boxes, binarized composite, finder quad,
  warped grid, sampled/classified matrices, palette swatches); observation
  only.
- **`diagroi.go`** - the ROI proposal and tile-map reports.

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

- **C-reference compatibility is the contract for the wire format** - today
  unconditionally, later as the C-compatibility side of the planned
  conformance-mode switch. Encoder output must be bit/format compatible with
  the reference C library so codes interoperate; the verified baseline is
  reference commit `3b56eef7` (2026-04-17). Ported functions name their C
  counterpart in a `// Ports ...` comment.
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
- **Determinism under concurrency.** Same input, same output, regardless of
  goroutine scheduling: banded pixel loops write disjoint rows, concurrent
  probe rungs write fixed result slots, and the resolution pyramid commits by
  fixed route priority, never first-done. Every pyramid route is a pure
  function of the input - the seeded route reads only the coarsest level's
  deterministic finding, published exactly once. Cancellation hooks only
  bound wasted work - they must never change the committed result.
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

*More than 8 colours.* The standard keeps colour modes beyond 8 as reserved
mode values (Annex G sketches 16-256-colour modes informatively only), but
the C library cannot actually handle them: its palette-placement table is
sized for at most 8 colours and it indexes out of bounds beyond that
(undefined behaviour). The port currently accepts only the 4- and 8-colour
modes (validation rejects other counts with an error). Support for the
higher modes is intended later; because no existing ecosystem reads such
codes, choosing them will come with explicit warnings.

*ECI / FNC1.* Decoding of these channels is only partially implemented, the
same as in the C reference.

### Known divergences from the C reference

Beyond the robustness extensions below, the port differs from the C library
in two deliberate ways:

- **Errors instead of undefined behaviour.** Where C indexes fixed tables
  with unvalidated input (unsupported colour counts, out-of-range ECC
  levels or docked positions, streams latching the unimplemented ECI/FNC1
  modes), the port validates first and returns an error - the never-panic
  invariant above. The wire format is unaffected.
- **Primary-retry re-binarization stays primary-scoped.** When the primary
  symbol needs the second, finder-seeded binarization pass, C overwrites the
  shared channel bitmaps, so its secondary (slave) detection runs on the
  re-binarized channels; the port keeps the swap local and detects
  secondaries on the first-pass channels. Observable only for a multi-symbol
  code whose primary needed the retry.

Everything else currently matches the C behaviour, including a couple of
decode quirks preserved verbatim; those are flagged with a "kept identical"
comment at their code sites. This blanket C parity is no longer an end in
itself: it is what the planned C-reference conformance mode keeps (likely as
the default), while the strict ISO/IEC 23634 mode planned alongside it will
follow the standard wherever the two disagree.

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
- **Region-of-interest retry** - as the last resort, the orientation search
  runs per proposed region (joint chroma-variance x gradient-energy tile
  score), restoring the module resolution a small symbol loses in the
  whole-frame probe downscale.
- **Two-regime module sampling** - small modules keep the C-ported 3x3
  centre kernel; larger modules are averaged over a tent-weighted central
  portion of their warped footprint, which suppresses screen-lattice ripple
  and sensor noise without reading neighbour-module smear at the edges.

Every scale-dependent value in these extensions (descreen kernel, binarization
block size, probe resolution, sampling footprint) is **estimated from the
image**. The one deliberate exception: contamination scales of optics and
codecs (defocus blur, JPEG chroma bleed, demosaicing) are physically fixed in
source pixels no matter how large a module appears, so the sampler's
small-module regime threshold is a pixel constant, with that justification
written at its definition.

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
