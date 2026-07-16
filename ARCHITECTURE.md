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

The default wire contract is the experimental target for
**ISO/IEC 23634:2022**. It uses the standard's 4-colour palette and placement
tables, rejects reserved colour modes, uses the Annex F generator for
interleaving and LDPC, and interprets the ISO message switches and ECI/FNC1
transmitted-data protocol, including the ISO/IEC 15434 message shift.
Independent validation of the Annex F range reduction remains before the
target can be promoted to verified strict conformance. The known differences
are listed under [Invariants](#invariants-and-cross-cutting-concerns).
On the decode side the port additionally goes **beyond** the C reference in
robustness - it reads rotated, screen-photographed and colour-cast captures the
C reader does not - without changing the wire format (see
[Robustness extensions](#robustness-extensions-beyond-the-c-reference)).

Optional build tags add decoder variants without changing the encoder default.
The reader carries them as a capability bitmask: ISO is always enabled, and
`jabcode_high_color`, `jabcode_bsi`, and `jabcode_legacy` each add their bit.
`Decode` automatically tries every enabled interpretation. Each raw,
average-RGB, descreen, or print finder pass traverses its prepared image once
and checks every compiled physical finder signature inside that traversal.
The current-family result is sampled once before ISO, high-colour and
current-C wire interpretations branch. ISO and ISO high-colour use one
representative observation because their 4- and 8-colour rules are identical;
a low-colour result is given the ISO identity after correction. When current-C
also needs correction, it reuses matching neutral module classifications and
soft reliability evidence while retaining its own mask, interleave, LDPC and
message rules. Alignment-pattern samples are reused only for exactly matching
interpreted geometry. The BSI/pre-v2.0 result likewise has one shared geometry
and sample. An untagged reader compiles the extra finder classifier and wire
branches out.
`jabcode_legacy` adds the read-only current and pre-v2.0 C-reference families,
including the older metadata, palette and recursive docked-secondary route.
`jabcode_bsi` adds exact BSI TR-03137 primary and docked-secondary decoding,
including the staged cross-edge metadata sample needed before a secondary's
full geometry is known. `jabcode_non_iso_encode` adds exact single- and
multi-symbol BSI output alongside the ISO-derived high-colour output. There is
no historical-C encoder.

The code is a small public package over a set of internal packages, plus thin
command-line front ends. The public API is deliberately small:

- `Encoder`, built with `NewEncoder(...Option)`, and its `Encode([]byte)`
  method (bytes to `image.Image`). Options: `WithColors`, `WithECCLevel`,
  `WithModuleSize`, and `WithSymbols`. The `jabcode_non_iso_encode` tag adds
  `Profile`, `ProfileISO23634`, `ProfileHighColor`, `ProfileBSI`, and
  `WithProfile` for selecting ISO, ISO-derived high-colour, or BSI output.
- `Decode(image.Image)` - image back to bytes under every compiled decoder
  capability. Forced single-variant decoding remains internal for CLI oracle
  work and tests.
- `Stream`, built with `NewStream()` - bounded `Decode` for one ordered,
  coherent frame sequence from any transport: it carries search hypotheses
  across frames and can
  combine compatible 4- and 8-colour primary evidence without entering the
  exhaustive single-image ladder. It automatically consumes every decoder
  capability compiled into the build through the same integrated finder and
  physical-family sampling graph as single-image decode.

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
- **`internal/diag`** - the passive diagnostic renderer behind
  `jabcode decode --diag`; renders the detailed observation trace returned by
  `internal/read` and never invokes a second decode pipeline.
- **`internal/ecc`** - LDPC construction/encode/decode (hard and soft),
  interleaving, and the fixed-seed PRNG they share.
- **`internal/wire`** - decoder wire variants and their additive capability
  bitmask, plus the separate single-format encoder choice propagated through
  palette, encoding and correction.
- **`internal/palette`** - module colour palette generation for all colour
  counts (4-256).
- **`internal/spec`** - symbol-layout constants and pure layout arithmetic
  (side sizes, metadata walk, mask values).
- **`internal/tables`** - the spec-derived constant tables (alignment
  positions, palette placement, colour-mode parameters, ...).
- **`internal/testutil`** - shared test-fixture access (central `testdata/`).
- **`cmd/jabcode`** - CLI wrapper over `Encoder`, `Decode`, and `diag`;
  `encode` reads payload bytes from stdin unless `--input` is set, `decode`
  writes payload bytes to stdout, and `decode --diag` writes the diagnostic
  report to stderr with optional annotated images under `--diag-out`.
  Untagged encoding is always ISO; `jabcode_non_iso_encode` adds the encoder
  `--profile iso|hc|bsi` selector. Decode always tries every compiled
  capability;
  its internal `--only` selector forces one compiled variant for oracle and
  debugging work. The ISO target remains experimental until its remaining
  validation closes.

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

The default build contains CPU and Vulkan preprocessing backends. For a frame
of at least 1024 by 1024 pixels, automatic selection lazily opens Vulkan once
and uses it only when the selected adapter reports a discrete-GPU device type;
smaller work, unavailable Vulkan, CPU implementations such as llvmpipe, and
currently unmeasured adapter classes use CPU. One retained GPU workspace is
leased to a decode and cached for another same-sized call. It uploads the
finest pyramid image once, derives every half-resolution level on the device,
and runs each upright level's complete raw, average-RGB, descreen and print
finder ladder against one detector state. Only packed binary masks, compact
finder-neighborhood and pitch reductions, and pixels required by confirmed
geometry, sampling or diagnostics cross back to the host. The CPU finder scan
and downstream geometry/decode remain authoritative consumers of those
outputs. Routes run concurrently: each leases a route context sized for its
canvas from the workspace pool, owning the rotation target, parameter buffer,
binding sets, resident binarizer and finder-pass preparer it mutates, while
the device, the read-only retained levels and the compiled kernels are shared.
One route's CPU scan therefore overlaps other routes' device kernels, and a
rotated canvas larger than the base frame gets a context of its own size
instead of falling back to CPU. Contexts are created on demand and reused;
exhausted device memory retires idle contexts and then waits for an in-flight
release, so it becomes backpressure rather than a failed route. A route that
encounters a genuine GPU error still falls back to the unchanged CPU route.
The coarse orientation probe and ROI proposal remain on CPU. CPU sampling and
decode after a GPU locate overlap the remaining resident operations.

`Decode` propagates its compiled capability bitmask through every route.
Within a route, every prepared image pass and row traversal happens once for
all compiled physical finder signatures. The located finding records which
physical family owns its quad, and seeded decoding preserves that family
instead of replaying detection. Geometry and finder-grid sampling happen once
per located family. ISO and ISO high-colour collapse to one current-family
observation; current-C remains a separate wire interpretation. Structurally
matching branches share neutral module classifications and soft reliabilities,
but each still applies its own mask, interleave, LDPC and message
interpretation.
Alignment-pattern sampling is cached by input version, side size and default
mode, so equal geometry is sampled once while genuinely different interpreted
geometry gets its own authoritative trace. The first successful interpretation
in fixed mask priority wins. Internal oracle helpers supply a one-bit mask and
therefore do not construct sharing caches. Diagnostics attach their trace to
that same single decode and never replay it under another variant.

After primary correction, every format enters one breadth-first docked-symbol
walk in `internal/read/docked.go`. The primary's established wire variant is
inherited by each secondary along with the host-decoded metadata seed. That
variant selects only the irreducible alignment-pattern recognizer, palette,
data-map and payload decoder; the secondary payload decoder recovers any
further docking metadata. Current-family and pre-v2.0 symbols share one
geometry implementation. Untagged builds compile a direct current-family
helper; the BSI-family alignment recognizer is present only when BSI or legacy
support needs it. BSI uses a staged branch inside the same walk: find the two
near alignment patterns, sample and decode the cross-edge metadata, then
locate the far pair and sample the complete secondary. It does not restart
whole-image detection.

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
  traverse docked symbols      read/docked.go, detect/detector_secondary.go,
  in breadth-first order       decode/decoder_secondary.go or tagged variant
```

As the last resort, the same orientation search runs per region of interest
(`detect/roi.go`, joint chroma-variance x gradient-energy tile score): a symbol
small within a large frame vanishes in the whole-frame probe downscale, and
probing the proposed region at its own scale restores the module resolution the
probe needs.

## Code map

### Public surface (root package)

- **`encoder.go`** - the `Encoder` type, functional `Option`s, input validation.
- **`profile_non_iso_encode.go`** - the build-tagged public encoder profile
  values and their internal format mapping.
- **`decode.go`** - the package-level `Decode` entry point.
- **`doc.go`** - package documentation.

### `internal/encode`

- **`encode_data.go`** - character analysis and the trellis that chooses the
  cheapest sequence of encoding modes, then emits the raw bit stream.
- **`encode.go`** - the single-symbol pipeline and bitmap -> image rendering,
  including `Render` (matrix + palette ground truth for tests/harness).
- **`bsi_enabled.go`, `bsi_multi.go`, `bsi_disabled.go`** - tagged exact BSI
  primary and docked-secondary encoder, multi-symbol data distribution and its
  untagged seam.
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
- **`detector_secondary.go`** - shared geometry and sampling of docked
  secondary symbols.
- **`bsi_family_primary.go`, `bsi_family_disabled.go`** - the build-tagged
  BSI/pre-v2.0 primary-finder classifier inside the shared row traversal and
  its compiled-out seam.
- **`pre_v2_c_secondary.go`** - the shared BSI-family alignment-pattern
  recognizer, with the pre-v2.0 tagged entry into the shared docked geometry.
- **`bsi_secondary.go`** - staged BSI secondary geometry: near patterns,
  cross-edge metadata sample, far patterns and complete-symbol sampling.
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
- **`pre_v2_c_primary.go`** - pre-v2.0 C-reference metadata, palette, data-map
  and primary/secondary payload decoding, compiled only with
  `jabcode_legacy`.
- **`bsi_primary.go`** - exact BSI TR-03137 primary metadata, palette,
  data-map and payload decoding, compiled only with `jabcode_bsi`.
- **`bsi_secondary.go`** - exact BSI secondary metadata, oriented palette,
  data-map and payload decoding, compiled only with `jabcode_bsi`.
- **`module_evidence.go`** - a fixed-size cache for neutral payload-module
  classifications and soft reliability evidence shared by compatible
  current-family wire interpretations. Variant-specific correction consumes
  copies and never mutates the retained neutral evidence.

### `internal/read`

- **`read.go`** - `Decode` and `DecodeImage`: the orientation and
  region-of-interest retries, the detect-then-decode primary handoff with the
  alignment-pattern fallback.
- **`docked.go`, `docked_variant_*.go`** - the one breadth-first
  docked-secondary graph walk, message assembly and build-tagged selection of
  current, BSI or pre-v2.0 secondary rules. The untagged selector is a direct
  current-family call.
- **`pre_v2_c_enabled.go`** - build-tagged read-only pre-v2.0 C-reference
  sampled-matrix interpretation.
- **`bsi_enabled.go`, `bsi_disabled.go`** - build-tag seam for the exact BSI
  sampled primary interpretation. Primary and recursively docked payloads use
  the common graph; BSI-only geometry and wire work remains behind the tag.
- **`historical_enabled.go`, `historical_disabled.go`** - geometry and
  sampling from the integrated detector's BSI/pre-v2.0 finder result. It runs
  once and branches to the enabled sampled-matrix interpretations.
- **`current_family_variants_*.go`** - build-tagged selection of the minimum
  irreducible current-family observations. ISO high-colour represents the ISO
  base when enabled; current-C remains a separate wire correction.
- **`alignment_observation.go`** - exact-geometry reuse of alignment-pattern
  sampling, including cached failures and one trace per actual sample.
- **`diagnostic.go`, `trace.go`** - the observation-only trace seam used by
  `DecodeWithTrace`; the normal and diagnostic entry points share the same
  route selection, sampling, metadata, palette and correction execution.
- **`pyramid.go`** - the resolution pyramid: level construction (one base
  conversion, box-halved levels above a shorter-side floor) and the
  concurrent per-level search with ordered commit (coarsest upright, seeded
  route, finer uprights, searches) and stage-boundary cancellation.
- **`seeded.go`** - the seeded route: re-enter the decode at the coarsest
  level's published finder quad and physical family on a finer level (scale,
  rotate, sample, decode - no fine finder search), committing on cross-scale
  byte agreement or, for a locate-only finding, on its own decode.
- **`stream.go`** - `Stream`: deterministic frame-sequence decoding under one
  replay, one upright scan, one carried rotated/finer attempt and one
  correction chain per frame. Recent winning quads and unused hypotheses are
  bounded search state; a miss never falls through to the exhaustive
  single-image ladder. One capability mask determines the finder signatures in
  the shared traversal; each physical family is sampled at most once per scan,
  and a deterministic cursor schedules only irreducible wire corrections
  across frames. The internal forced-variant constructor uses the same graph.
- **`evidencegroup.go`** - the separate fixed-anchor content state for 4- and
  8-colour primary symbols: deeply owned observations, reject-only layout and
  wire-variant and spatial compatibility, bounded signed evidence,
  mixed-content rejection, material-change correction scheduling and confirmed
  canonical signatures. Search geometry and content evidence have separate
  lifetimes. Non-ISO variants never enter the ISO evidence accumulator.

The integrated Stream route is:

```text
ordered frame
  -> prepared canvas
  -> one finder-row traversal using compiled signatures
       -> current signature -> one sample -> ISO/high-colour/current-C rules
       -> BSI signature     -> one sample -> BSI/pre-v2.0 rules
  -> deterministic route cursor
  -> at most one payload correction
  -> payload or bounded miss
```

Disabled signatures and their interpretation branches compile out. A transport
only has to supply coherent frames in order; it is not part of the reader.

### `internal/diag`

- **`diag.go`** - `Diagnose`: runs `read.DecodeWithTrace` once and reports its
  final payload or error.
- **`diagtrace.go`** - renders the returned route, probe, detector, sampling,
  palette, correction, alignment and secondary observations without rerunning
  any of them.
- **`diagimg.go`** - the per-stage annotated image sink behind `Diagnose`'s
  image-directory mode. It shows the untouched input and pyramid levels,
  every orientation probe, separate region feature maps, detector retry
  inputs, finder and secondary geometry, warped grids, alignment patterns,
  print-aware channel sample positions, metadata and palette walks, payload
  layout, sampled/classified comparisons and palette swatches. Reserved
  modules that production decode does not classify are classified only while
  rendering their diagnostic image. Multiple real alignment samples receive
  distinct numbered overlays, and reused neutral classification is labeled in
  the trace. Docked traces name the established wire variant, and tagged
  pre-v2.0 secondaries retain their authoritative payload classification from
  the same decode. The sink remains observation only.

### `internal/ecc`

- **`ldpc.go`** - LDPC code construction (Gallager + Gauss-Jordan), encoding,
  hard-decision decoding and read-only syndrome measurement.
  **`ldpc_soft.go`** - log-domain belief propagation for the single-frame
  fallback plus a direct signed-channel entry for accumulated evidence.
  **`bitmatrix.go`** - dense GF(2) matrix.
- **`interleave.go`**, **`random.go`** - fixed-seed PRNG-driven (de)interleaving.

### Commands and fixtures

- **`cmd/jabcode`** - CLI with `encode` and `decode` subcommands; diagnostics
  are part of `decode --diag`.
- **`testdata/`** - golden vectors (bit streams, matrices, palettes) checked
  against the C reference, clean C-encoded fixtures, and the detection
  snapshot golden; consumed via `internal/testutil`.

## Invariants and cross-cutting concerns

These hold across the whole module; breaking one is an architectural change, not
a local one.

- **ISO encoding, additive decoding.** A default `Encoder` writes the
  experimental ISO target. An untagged `Decode` accepts only ISO; build tags
  add high-colour, BSI and legacy decoder bits. The fixed attempt order is ISO,
  high-colour when ISO rejected a reserved colour mode, current C, BSI, then
  pre-v2.0 C. Successful ISO reads never pay a fallback correction chain. The
  verified C-reference baseline is commit `3b56eef7` (2026-04-17). Ported
  functions name their C counterpart in a `// Ports ...` comment.
- **Naming: primary/secondary.** The reference C library calls the two symbol
  roles "master"/"slave"; this port uses **primary**/**secondary** throughout
  (types, functions, files). Comments bridge to the old C names where helpful.
- **Determinism via fixed PRNG seeds.** Interleaving and LDPC matrix
  construction use the standard's fixed seeds (data-stream LDPC, metadata
  LDPC, and interleaving each have their own). The PRNG lives in
  `internal/ecc/random.go`. These seeds are part of the wire format - do not
  change them. The selected profile chooses either the C/BSI generator or the
  ISO Annex F generator; both are deterministic and have separate LDPC cache
  keys.
- **Determinism under concurrency.** Same input, same output, regardless of
  goroutine scheduling: banded pixel loops write disjoint rows, concurrent
  probe rungs write fixed result slots, and the resolution pyramid commits by
  fixed route priority, never first-done. Within one level's search the
  orientation rungs and region retries also run concurrently and commit in
  ladder order through the same fixed-slot discipline, so the winning route
  and the published finding match the sequential ladder exactly. Every route
  is a pure function of the input - the seeded route reads only the coarsest
  level's deterministic finding, published exactly once. Cancellation hooks
  only bound wasted work - they must never change the committed result.
  One scoping caveat: rotated-route GPU output is not bit-identical to the
  CPU reference, and which backend a route uses is not fully pinned - the
  process-wide workspace is leased to one decode at a time (concurrent
  `Decode` calls race for it), and a route that finds device memory
  exhausted, whether by other processes or by which sibling routes
  allocated first, falls back to CPU. A borderline rotated capture can
  therefore in principle resolve differently between runs whenever the
  device is contended or memory-pressured, even for a single serial
  `Decode`. Hosts without a qualifying GPU are unaffected.
- **Colour-mode scope.** ISO accepts only the normative 4- and 8-colour modes
  and rejects reserved `Nc` values. The tagged high-colour profile is the
  ISO-derived 16- through 256-colour extension. Legacy accepts current-C 4- and
  8-colour symbols plus this port's historical extension fixtures; BSI has its
  own specified 4- through 256-colour layouts. Validation happens before
  profile-specific tables are indexed, so malformed input returns an error
  rather than panicking.
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

### Wire profiles and ISO differences

*4-colour palette order.* The standard (Tables 4/21) orders the four colours
black, cyan, magenta, yellow; the legacy C family uses black, magenta, yellow,
cyan; BSI TR-03137 uses blue, green, magenta, yellow. ISO uses the standard's
order together with its finder, alignment and embedded-palette indices.
Correction and interleaving also differ, so the additive reader verifies each
enabled wire interpretation instead of inferring it from palette order alone.

*More than 8 colours.* The normative standard keeps colour modes beyond 8 as
reserved mode values; ISO/IEC 23634 Annex G (informative) specifies the
16-256-colour palettes and their embedding. The C reference cannot handle these
modes: its palette-placement table is sized for eight colours and indexes out of
bounds beyond that (undefined behaviour); its normalized-RGB module classifier
collapses colours that share a hue but differ in brightness, exactly the
intermediate levels a larger palette introduces; and it embeds the palette in
four copies, where Annex G calls for two ("128 modules reserved for two colour
palettes"). Finder detection, by contrast, is mode-independent: the finder cores
carry the same physical colours in every mode.

The tagged high-colour profile adds working 16-, 32-, 64-, 128- and
256-colour modes as a deliberate, non-standard ISO-derived extension following
Annex G. ISO rejects those reserved modes. Four structural changes are gated on
the colour count so its 4- and 8-colour paths stay identical to ISO:

- **Two palette copies, not four** (`spec.PaletteCopies`). Four copies of the
  up-to-64 embedded colours (64x4 = 256 placement modules) overflow the primary
  symbol's fixed metadata region - its module walk defines distinct positions only
  to roughly 172 before repeating - whereas two copies (up to 64x2 = 128) fit, as
  Annex G intends ("128 modules reserved for two colour palettes").
- **Every embedded colour in the metadata region, none in the finder**
  (`spec.PaletteFinderColors` returns 0 above 8 colours). The 4/8-colour
  layout carries palette colours 0 and 1 in the
  finder/alignment cores, but those cores are not palette colours 0 and 1 in the
  higher modes, so reading them there corrupts two entries - and, once 128/256
  interpolate from them, several more. Annex G is explicit: "all available colours
  should be included in the embedded colour palettes."
- **Absolute-RGB classification** against the embedded palette, preserving the
  brightness the reference's normalized-RGB match discards (it collapses colours
  that share a hue but differ in brightness - exactly the intermediate levels a
  larger palette introduces).
- **Part I colour-mode marker at each mode's own black/cyan/yellow index**
  (`tables.NcMetadataColorIndex`); the reference's fixed 0/3/6 render as unrelated
  colours in the larger grids, so Part I could not be read back before the palette.

With every embedded colour captured exactly and the interpolation
(`interpolatePalette`, for 128/256) reconstructing the rest from correct anchors,
all counts round-trip across a payload sweep on pixel-exact synthetic input.
Physical robustness shrinks with the palette and is measured on the committed
real-capture set (`testdata/highcolor_capture`, frontal well-lit captures at
maximum ECC): a phone camera photographing a display reads 16 colours reliably
and 32 marginally, the same camera on a laser print reads up to 32, a flatbed
scan reads up to 128, and 256 decodes only pixel-exact digital images. The
measured mechanism is classification density, not geometry: on verified
true-grid samples, the data-module bit error rate against the re-encoded
ground truth crosses what the strongest LDPC setting (wc/wr 6/7, soft retry
included) corrects at roughly 7 percent - decoding capture rows measure
0.063-0.069, the nearest failing camera rows 0.072-0.086, and display
128c/256c reach 0.19-0.21. The higher palettes pack the RGB cube to about one
quantisation step between neighbours (256 colours on roughly 8x8x4 levels), so
capture noise and illumination casts collapse adjacent colours. Treat 64c+ as
a lossless synthetic container or at most scanner-grade codes, not
camera-scannable ones. Finder detection needs no change - the finder
cores carry the same physical colours in every mode. The palette-placement order
is the identity beyond the eight reference-shuffled slots, shared by encoder and
decoder. These codes are not interoperable - no other decoder reads them (the
reference is broken for the reasons above) - so the CLI warns on stderr and
`WithColors` documents it. A current-family docked secondary caps at 32
colours, the size of its palette-position table. BSI uses its separate
two-palette secondary layout and carries up to 64 representatives for all
specified colour modes through 256.

*Message controls, ECI and FNC1.* The current-C and pre-v2.0 C variants retain
the reference decoder's partial-message behavior when their unimplemented
ECI/FNC1 sentinel modes are reached. The ISO variant instead follows Tables 14
to 19, including the lowercase numeric shift, URL shortcuts and all three ECI
assignment widths. Its
transmitted data always uses the required Annex H `]jN` identifier, escapes ECI
assignments and literal backslashes, and emits in-mode FNC1 as the ASCII GS
separator. Because this variant models an ECI-capable reader, every successful
transmission begins with `]j1`, `]j4` or `]j5`, even when the message contains
no explicit ECI assignment. Malformed, reserved and
unterminated controls reject the route. The ISO/IEC 15434 switch contributes
its `[)>` plus RS message header and stays active until the JAB EOT control.
That control normally contributes the EOT message trailer; formats `02` and
`08` suppress it because ISO/IEC 15434 makes each an exclusive whole-message
format without RS or EOT trailers. This state is separate from and mutually
exclusive with FNC1. It validates macro framing and the decimal format
indicator, not application data inside the format envelope. The encoder does
not yet offer structured input for emitting these optional controls.

*Default byte charset.* ISO/IEC 23634 (5.3.1) interprets byte-mode data as
UTF-8 (ISO/IEC 10646); the pre-ISO BSI TR-03137 specified ISO/IEC 8859-15
(ECI 000017). The byte-mode wire encoding is identical raw 8-bit values either
way, and the decoder emits those bytes unchanged and leaves the charset to the
caller (as the C reference does), so this is a documented-semantics delta, not
a byte-stream one.

*Pseudo-random generator.* Interleaving and LDPC matrix construction are
driven by a seeded PRNG. ISO/IEC 23634 Annex F specifies the ISO/IEC 9899
(ANSI C) `rand` (`next = next*1103515245 + 12345`, RAND_MAX 32767); the
current-C, pre-v2.0 C, and BSI variants use the reference library's 64-bit LCG
(`s = 6364136223846793005*s + 1`) with MT-style output tempering, matching
BSI TR-03137 Annex E. The ISO variant uses the Annex F generator for
interleaving and message and metadata LDPC. Annex F asks for an index in the
current range but does not state the reduction operation; this implementation
uses `floor(rand()/32768*range)`, preserving the reference algorithm's scaling
shape. The generator, range mapping and seeds live in `internal/ecc/random.go`.

### Known divergences from the C reference

Beyond the robustness extensions below, the port differs from the C library
in two deliberate ways:

- **Errors instead of undefined behaviour.** Where C indexes fixed tables
  with unvalidated input (unsupported colour counts, out-of-range ECC levels
  or docked positions), the port validates first and returns an error - the
  never-panic invariant above. The current-C and pre-v2.0 C variants retain the
  reader's ECI/FNC1 partial-message behavior without indexing outside the
  character-size table; the ISO variant interprets those controls as described
  above. The wire format is unaffected.
- **Primary-retry re-binarization stays primary-scoped.** When the primary
  symbol needs the second, finder-seeded binarization pass, C overwrites the
  shared channel bitmaps, so its secondary (slave) detection runs on the
  re-binarized channels; the port keeps the swap local and detects
  secondaries on the first-pass channels. Observable only for a multi-symbol
  code whose primary needed the retry.

Everything else in the current-C variant matches C
behaviour, including a couple of decode quirks preserved verbatim; those are
flagged with a "kept
identical" comment at their code sites. ISO is the default encoder format and
the always-present first decoder capability.

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
- **The `jabcode decode --diag` diagnostic** runs the authoritative decoder
  once on a real capture and renders its observation trace with full evidence
  dumps for measure-first debugging.

The test files alongside the sources cover round-trips (Go encode -> Go decode),
decoding C-produced vectors, multi-symbol cascades, soft- and hard-decision LDPC,
rotation recovery, and option validation.

## References

- ISO/IEC 23634:2022 - *Information technology - Automatic identification and
  data capture techniques - JAB Code polychrome bar code symbology specification*.
- Reference implementation: [github.com/jabcode/jabcode][jabcode].

[jabcode]: https://github.com/jabcode/jabcode
