# JAB Code wire format

Reference for the JAB Code wire format as implemented by the C library and this
Go port, with the known ISO and pre-ISO (BSI) deltas. Read this before
re-reading a spec or C source; update it whenever a spec, C-history, or
compatibility fact is checked.

Confidence convention: entries are verified against the current C reference and
this Go port unless marked otherwise. `[ISO]` is from ISO/IEC 23634:2022; `[BSI]`
is from BSI TR-03137 Part 2.

## Baseline

- C reference: github.com/jabcode/jabcode at commit `3b56eef7` (2026-04-17). The
  Go code and golden tests were ported against this lineage. Record whether any
  upstream C change is wire-format relevant before advancing this commit.
- ISO/IEC 23634:2022, the published standard. Clause and table numbers below are
  ISO's.
- Pre-ISO JAB Code spec: BSI TR-03137 **Part 2** (it has a text layer - the cheap
  route into the pre-ISO format). BSI TR-03137 **Part 1** is the Digital Seal
  application document (digital seals over DataMatrix and profile encodings), not
  the JAB Code symbology, so it is not a wire-format source.

## Contract

An untagged `Encoder` writes the experimental ISO target and exports no format
selector. `jabcode_non_iso_encode` adds the public `Profile` type,
`ProfileISO23634`, `ProfileHighColor`, and `WithProfile`; it currently exposes
only ISO and the ISO-derived high-colour extension. BSI output remains internal
until its primary and docked-secondary implementation is complete. No encoder
emits either historical C variant.

Decoding is additive. The reader carries a capability bitmask with ISO always
set; `jabcode_high_color` adds ISO high-colour, `jabcode_bsi` adds BSI, and
`jabcode_legacy` adds both current-C and pre-v2.0 C. `Decode` and ordinary CLI
decode try every compiled capability in fixed order. Forced single-variant
decoding is internal and exposed only through the CLI `--only` oracle/debugging
flag and tests. Current-family image preparation, finder detection, and grid
sampling are shared before ISO, high-colour, and current-C wire interpretations
branch. ISO high-colour represents the common ISO base when enabled because
its 4- and 8-colour rules are identical to ISO; successful low-colour results
are labeled ISO without another correction chain. A structurally matching
current-C interpretation reuses neutral module classifications and soft
reliabilities, but applies its own masking, interleaving, LDPC and message
rules. Alignment-pattern samples are shared only when input version, side size
and default-mode state match exactly. The current and BSI/pre-v2.0 physical
signatures are checked inside the same row traversal of each raw, average-RGB,
descreen, or print pass. BSI and pre-v2.0 C then share one geometry and sample
from their common finder result. Seeded decoding retains the finder family and
samples known geometry once instead of re-detecting it. This is one
image-search pipeline, not a full decode replay per variant. Disabled
signature classifiers compile out.

The tagged historical-C reader handles both current-C and pre-v2.0 C symbols and
recursively traverses their docked secondaries. No encoder emits the historical
C formats. The BSI decoder tag currently adds exact primary-symbol decoding.
Exact primary-symbol encoding is verified module-for-module against Annex C
but remains internal; public BSI availability waits for its different
docked-secondary layout. The ISO variant remains an experimental target, not a
verified strict-conformance claim, until independent Annex F validation closes.

The ISO variant currently covers the 4-color palette and its fixed pattern and
palette placements, reserved color modes, the Annex F generator, interleaving,
message and metadata LDPC construction, Table 14 to Table 19 mode switches, and
the ISO/IEC 15434, ECI and FNC1 transmitted-data protocols. Annex F leaves the
reduction from `rand()` to a requested range unstated; the deterministic
interpretation used here is documented below.

Robustness changes in detection, sampling, and retry order are allowed when they
do not change clean-case output. Frozen wire format: encoder bit layout, palette
indices, metadata layout, masks, PRNG seeds, interleaving, and LDPC
construction.

## Color palettes

- 8-color palette is the RGB cube: black, blue, green, cyan, red, magenta,
  yellow, white. Matches ISO Table 3 `[ISO]`; no ordering divergence.
- 4-color palette diverges across families. Current C uses black, magenta,
  yellow, cyan (8-color indices `[0, 5, 6, 3]`); ISO Table 4 `[ISO]` orders them
  black, cyan, magenta, yellow; BSI uses blue, green, magenta, yellow. Each
  profile uses its corresponding finder, alignment and embedded-palette indices.
  The additive
  reader verifies enabled interpretations instead of inferring the whole wire
  format from palette order.
- C: `encoder.h` `jab_default_palette`, `encoder.c` `setDefaultPalette`.
  Go: `internal/palette.Default`, `internal/palette.SetDefault`.

## Color modes above 8

The metadata `Nc` field can name modes for 16/32/64/128/256 colors, but the
current C reference cannot read or write them soundly.

- ISO/IEC 23634:2022 Table 6 defines only mode 1 (4 colors) and mode 2 (8
  colors); modes 0 and 3-7 are reserved (Annex G is informative only) `[ISO]`.
  Pre-ISO BSI TR-03137 defines modes 3-7 normatively as 16/32/64/128/256 colors
  `[BSI]` (Table 5).
- C palette-placement tables hold 8 columns
  (`master_palette_placement_index[4][8]`, `slave_palette_placement_index[8]`)
  but both encoder and decoder iterate up to `min(color_number, 64)`, so modes
  above 8 index out of bounds. `genColorPalette` (16-256) and
  `interpolatePalette` (128/256) exist, but the table overflow makes the path
  unsound.
- The tagged high-colour profile accepts 4, 8, 16, 32, 64, 128 and 256 colors
  for a single symbol (a multi-symbol code caps at 32). Its 4 and 8 modes match
  ISO; above 8 it follows ISO Annex G - two palette copies, every
  color embedded up to 64 (128/256 embed those 64 representatives and interpolate
  the rest on decode), classified in absolute RGB. These higher modes are a non-interoperable,
  digital-only extension (see ARCHITECTURE.md); no other decoder reads them.
- The committed `testdata/highcolor_capture` set predates that ISO-derived
  profile. Its 16- through 256-color symbols use the current-C generator,
  interleaving, LDPC and message controls with the historical high-color
  palette extension, so its decoder and harness require `jabcode_legacy`.
  It is physical-capture evidence for `CurrentC`, not an ISO-high-color wire
  oracle.
- ISO mode rejects encode modes above 8 and rejects their reserved `Nc` values
  on decode, following normative Table 6 rather than informative Annex G.
- Go: `internal/tables.PrimaryPalettePlacement`,
  `internal/tables.SecondaryPalettePlacement`,
  `internal/decode.ReadColorPaletteInPrimary`,
  `internal/palette.genColorPalette`, validation around `WithColors`.

## Metadata layout

### Current format (C >= v2.0, this port = ISO/IEC 23634:2022)

The current C and Go layout is the ISO layout (ISO Table 5) `[ISO]`. Primary
metadata is three parts:

| Part | Content | Raw | Enc | Location |
| ---- | ------- | --- | --- | -------- |
| I | `Nc` module color mode | 3 | 6 | metadata modules |
| II | `V`(10), `E`(6), `MSK`(3) | 19 | 38 | metadata modules |
| III | `S` docked positions | 4 | with data | data stream |

Part I and Part II are optional: a default primary symbol omits them and the
decoder loads defaults (8 colors, mask reference 7, ECC level 3, side version
from the sampled matrix size). `V` holds the horizontal side-version in its
first 5 bits and the vertical in its last 5 (side-version x as `x-1`). Part I is
encoded three-color in black, cyan, and yellow (default-palette indices 0/3/6,
via `tables.NcColorEncode`) - its four modules read as two color-pairs give the 6
encoded bits (ISO Table 7); the rest of the metadata uses all palette colors. The
embedded palette follows Part I.

Part III (`S`) is not in the metadata modules: it is appended at the end of the
data stream in bitwise-reversed order, followed by a flag bit set to binary one,
with any docked secondaries' metadata placed between the message and `S`. That
trailing flag is the start-flag the Go stream parser scans for (and bounds; C's
scan is unbounded on all-zero streams).

- C: `decoder.c` `decodeMaster`, `readColorPaletteInMaster`.
  Go: `internal/spec` (`PrimaryMetadataPart1Length` 6,
  `PrimaryMetadataPart2Length` 38, `PrimaryMetadataPart1ModuleNumber` 4),
  `internal/decode/decsym.go`, `internal/tables.NcColorEncode`.
- The Part I finder-core color-reference retry in Go is a robustness extension;
  it does not change the wire layout.

### Secondary symbols (current)

A docked secondary symbol carries its own three-part metadata (ISO Table 9): Part
I flags `SS` (same shape/size as host) and `SE` (same ECC as host), 2 bits; Part
II `V` (0 or 5 bits) and `E` (0 or 6); Part III `S` (docking of the three free
sides, 3 bits); total 5-16 bits. Parts I and II ride the host symbol's data
stream (ahead of the host's own `S`); Part III rides the secondary's own data
stream. A secondary's first two palette colors come from alignment-pattern
positions rather than finder cores (see Palette placement). Go:
`internal/decode/decoder_secondary.go`.

### Pre-ISO format (C < v2.0 and BSI TR-03137)

Pre-ISO metadata is three-part but variable-length and flag-driven: flags in
Part II (`SS`, `VF`, `SF`) select the lengths of `V`/`E` and signal shape and
docking. `d315eb9` (2019-09-10, "Version 2.0.0 - new metadata structure, new
color palettes, new finder/alignment color combination") replaced it with the
fixed ISO layout above (`V` always 10 bits, `E` 6, no `SS`/`VF`/`SF` flags, `S`
moved to the data stream), so a pre-ISO decoder needs its own metadata walk, not
just different constants.

BSI TR-03137 Part 2 primary metadata (Tables 4-7) `[BSI]`:

| Part | Variables (raw bits) | Raw | Encoded |
| ---- | -------------------- | --- | ------- |
| I | `Nc` (3) | 3 | 6 |
| II | `SS` (1), `VF` (2), `MSK` (3), `SF` (1) | 7 | 14 |
| III | `V` (2-10), `E` (10-16), `S` (0 or 4) | 12-30 | 24-60 |

`V` length depends on `SS`+`VF`, `E` on `VF`, `S` on `SF`. Total 22-40 raw,
44-80 encoded. Secondary metadata is likewise three-part (Tables 8-11): Part I
`SS`/`SE`/`SF` (3 raw), Part II `V` (0/5) and `S` (0/3), Part III `E`
(0 or 10-16); total 3-27 raw, 6-54 encoded. (Part 2's per-table "total length"
rows read 14-42 and 3-30, which disagree with the field sums above and with the
prose in section 3.4; the field sums are the internally consistent figures.)

BSI also encodes Part I `Nc` differently: two-color mode using the palette's
first and last colors (black/white, or blue/yellow in the 4-color mode 001), not
ISO's three-color black/cyan/yellow (BSI 3.4.1.1). A pre-ISO Part I reader
differs in module colors as well as field lengths.

Pre-v2.0 C confirms a three-part walk but does not match BSI field-for-field:
at `2ece74e`, `decodeMaster` reads Part I `Nc` (6 encoded), Part II `SS`+`VF`+`MSK`
(12 encoded, no `SF`), and Part III `V`+`E` (`MASTER_METADATA_PART3_MAX_LENGTH`,
16-32 encoded, no `S`). Full pre-ISO read support therefore means reading the
pre-v2.0 C directly, not just the BSI text.

Beyond metadata, `d315eb9` also changed (verified in its `encoder.h` diff): it
introduced the palette-placement tables (`master_palette_placement_index`,
`slave_palette_placement_index` did not exist before it); it changed the
finder-pattern core colors from `{blue, green, magenta, yellow}` (`FP0..3` =
1,2,5,6) to `{black, black, yellow, cyan}` (0,0,6,3), and rewrote the
per-color-mode `fp*_core_color_index` and `ap*_core_color_index` tables. Pre-ISO
alignment patterns are monochrome (`[BSI]`: U/L white outer with black core, X0/X1
black outer with white core) versus the current cyan/yellow. The 8-color default
palette cube values are unchanged. So pre-ISO read support is a
broader change than the metadata walk alone.

BSI Part 2 is also inconsistent on data-module encoding: its prose (like the
current format) uses the data value as the palette index directly, but its Table
19 remaps values to colors (0 black, 1 magenta, 2 yellow, 3 cyan, 4 red, 5 green,
6 blue, 7 white) - an order that also differs from its own palette cube (Table
3). A pre-ISO reader has to resolve which BSI actually intends; the current
format is unambiguous identity (value = palette index).

## Message controls, ECI and FNC1

ISO specifies message controls that differ from the C reference in two places.
The C decoder treats uppercase value 31 plus `11` as end of message, while ISO
Table 15 reads another three-bit switch. The C decoder treats lowercase value 31
plus `11` as its unimplemented FNC1 sentinel, while ISO Table 16 defines a
one-character numeric shift. The current-C and pre-v2.0 C variants preserve
both reference behaviors. The ISO variant follows the tables and also expands
the `https://`, `http://`
and `www.` shortcuts.

ISO ECI assignments use the 8-, 16- and 22-bit forms from Table 19. Decode
transmits each assignment as a backslash followed by six decimal digits and
doubles every literal data backslash. The ISO variant models an ECI-capable
reader, so every transmission carries the required Annex H symbology
identifier: `]j1` for an ordinary message, `]j4` for FNC1 before the first
message character, and `]j5` for FNC1 after one initial letter or two initial
digits. The `]j0`, `]j2` and `]j3` modifiers describe an ECI-disabled reader
mode that this API does not expose. An in-mode FNC1 is transmitted as ASCII GS
(29), and EOT ends the FNC1 mode without being transmitted.

The Table 15 ISO/IEC 15434 switch represents the four-byte message header
`[)>RS`, where RS is byte 30, and starts a message-format shift. The next two
data bytes are its decimal format indicator. The JAB EOT control ends the shift
and normally transmits the ISO/IEC 15434 EOT message trailer, byte 4. Formats
`02` and `08` are exclusive whole-message formats that forbid both the RS
format trailer and the EOT message trailer, so their JAB EOT control ends the
shift without transmitting a byte. The decoder does not synthesize format
trailers; required RS bytes remain part of the encoded data. A literal byte 4,
including one inside format `09` binary data, remains data and does not end the
shift.

The ISO route validates the JAB macro boundary and two-digit format indicator,
not the application syntax inside an ISO/IEC 15434 format envelope. A misplaced,
nested, repeated, truncated or unterminated message-format control is rejected,
as is mixing the ISO/IEC 15434 and FNC1 message protocols. Truncated ECI
assignments, reserved switches, invalid FNC1 placement and a missing or stray
EOT are also rejected. The current-C and pre-v2.0 C variants still stop cleanly
at their unimplemented ECI/FNC1 sentinels, matching the reference decoder's
partial-message behavior without indexing beyond the character-size table. The
encoder does not yet expose structured input for emitting ECI, FNC1 or ISO/IEC
15434 controls.

The default byte-mode interpretation also differs at the spec level: ISO/IEC
23634 (5.3.1) specifies UTF-8 (ISO/IEC 10646); BSI TR-03137 specifies ISO/IEC
8859-15 (ECI 000017). The byte-mode wire encoding is identical (raw 8-bit
values) and the decoder emits those bytes, leaving the charset to the consumer,
so this is a documented-semantics delta, not a byte-stream one.

- C: `decoder.c` `decodeData`. Go: `internal/decode/decoder.go` `DecodeData`
  and `DecodeDataProfile`.

## Palette placement

For 4- and 8-color symbols, both primary and secondary symbols embed four palette
copies. In the primary, each copy's first two colors are read from modules inside
the finder patterns and the rest along the metadata/palette walk. In secondaries,
the first two come from alignment-pattern positions and the rest from fixed
secondary palette positions. Go mirrors the C placement tables here.

Above 8 colors Go follows ISO Annex G instead of the C tables (which overflow,
see above): **two** palette copies, and all colors up to 64 embedded in the
metadata region (the finder cores are not palette colors 0 and 1 in these
modes); 128/256 embed those 64 representatives and interpolate the rest.
Pre-ISO BSI TR-03137 embeds two palette copies as well (up to 128 reserved
palette modules), a distinct format `[BSI]`.

- C: `master_palette_placement_index`, `slave_palette_placement_index`,
  `slave_palette_position`, `readColorPaletteInMaster`,
  `readColorPaletteInSlave`.
  Go: `internal/tables.PrimaryPalettePlacement`,
  `internal/tables.SecondaryPalettePlacement`,
  `internal/tables.SecondaryPalettePosition`,
  `internal/decode/paldecode.go`, `internal/decode/decoder_secondary.go`.

## Finder and alignment patterns

- **Quiet zone**: none required. ISO/IEC 23634 4.3 states explicitly, for
  square and rectangle primary and secondary symbols alike, that no quiet zone
  surrounding the symbol is required (finder cores are colour-structured, not
  margin-dependent) `[ISO]`.
- **Finder patterns**: four, one per corner, each built from square references of
  three one-module layers in two alternating colors (ISO 4.3.7). UL and LL are
  black/cyan, UR and LR are yellow/black; cores are UL/UR black, LR yellow, LL
  cyan - i.e. FP0/FP1 black (0), FP2 yellow (6), FP3 cyan (3) in the default
  palette, scaled for higher color modes per `tables.FPCoreColor`. Go:
  `internal/spec` (`FP0CoreColor`..`FP3CoreColor`), `internal/tables.FPCoreColor`,
  `internal/detect`.
- **Alignment patterns**: present from Side-Version 6 up. Four types: U and L
  have a yellow outer layer with cyan cores (core 3), X0 and X1 a cyan outer
  layer with yellow cores (core 6); higher modes scale per `tables.APNCoreColor`
  (U/L) and `tables.APXCoreColor` (X0/X1). Positions follow ISO Table 2 and are
  ported from the C reference. Go: `internal/tables`, `internal/detect`.

## Preserved C quirks

Two C quirks are kept identical in Go because they affect decode behavior:

- Reconstructed secondary alignment pattern `t4`: its module-size estimate
  averages `t1` twice and omits `t2` (the position math uses the correct
  parallelogram form; only the size expression is off). C `detector.c`
  `findSlaveSymbol`, Go `internal/detect/detector_secondary.go`.
- Secondary palette threshold setup offsets the palette copy by `i*3` rather than
  `colorNumber*3*i` (the primary path uses the correct `colorNumber*3*i`).
  C `decoder.c` `decodeSlave`, Go `internal/decode/decoder_secondary.go`.

## Masks, PRNG, interleaving, LDPC, side versions

- **Data masks** (ISO Table 22): eight patterns. The mask value `m(x,y)` is XORed
  into the data module's color index, taken mod the color count; only data
  modules are masked. Generators: 0 `x+y`; 1 `x`; 2 `y`; 3 `x/2 + y/3`;
  4 `x/3 + y/2`; 5 `(x+y)/2 + (x+y)/3`; 6 `(x*x*y)%7 + (2*x*x+2*y)%19`;
  7 `(x*y*y)%5 + (2*x+y*y)%13`. Default reference 7; selection minimizes three
  penalty rules (ISO Table 23). Go: `internal/spec.MaskValue`,
  `internal/encode/mask.go`.
- **PRNG**: current C, the Go current-C and pre-v2.0 C variants, and BSI use a
  64-bit LCG
  `s = 6364136223846793005*s + 1`
  with MT-style output tempering (matching BSI Part 2 Annex E), seeded `785465`
  (message LDPC), `38545` (metadata LDPC), `226759` (interleaving). The seeds are
  ISO's too, but the generator is not: ISO/IEC 23634 Annex F specifies the
  ISO/IEC 9899 (ANSI C) `rand` (`next*1103515245 + 12345`, RAND_MAX 32767).
  The ISO variant uses that generator with 32-bit low-word arithmetic for
  interleaving and both LDPC matrix seeds. Annex F requires each permutation
  index to be in `[0,L)` but does not state how to reduce the 32768 possible
  `rand()` results to that range. This implementation retains the reference
  algorithm's scaling interpretation, `floor(rand()/32768*L)`, rather than
  using modulo. That choice is pinned by tests but still needs an independent
  strict-ISO oracle. Go: `internal/ecc/random.go`, `ldpc.go`, `interleave.go`.
- **LDPC**: a systematic generator matrix built over GF(2) from the seeded PRNG;
  default ECC level 3 (`wc=4, wr=9`; BSI TR-03137 defaults to level 6). Go:
  `internal/ecc`.
- **Side versions**: 32 sizes, side = `version*4 + 17` modules (21 at v1 to 145
  at v32), with independent horizontal/vertical versions for rectangles. Go:
  `internal/spec.VersionToSize`.

### ISO Annex D example caveat

The Annex D example is not a strict-ISO wire oracle. Its message is
`JAB Code 2016!`, but informative Table D.1 assigns digit `6` numeric value 5
and bits `0101`; normative Table 13 assigns it value 7 and bits `0111`. The
implementation follows Table 13.

The Figure D.1 raster was extracted directly from the native PDF page image and
sampled as a 21 by 21 module matrix, avoiding OCR and resampling of the printed
figure. All 441 modules match the current C-reference encoder's output for the
message using the normative digit value. The figure therefore also uses the
C/BSI generator rather than Annex F. It is a useful current-C/legacy golden,
but it cannot validate the ISO PRNG or settle Annex F's missing range-reduction
rule.

## Go differences that do not change the wire format

Intentional implementation differences, documented in `ARCHITECTURE.md` and at
their code sites:

- Unsupported color counts and invalid parameters return errors instead of
  indexing fixed tables out of bounds.
- Current-C and pre-v2.0 C ECI/FNC1 handling checks the sentinel mode before
  indexing the character-size table, while the ISO variant interprets the
  controls directly; neither path can panic on those mode values.
- The hard-LDPC data path checks the post-correction syndrome and reports failure
  instead of returning a corrupt payload with `err == nil`.
- Primary-retry re-binarization is primary-scoped in Go; in C the retry writes
  through the caller's channel array, so secondary detection can see the
  re-binarized channels. Detection behavior, not wire format, and observable only
  for multi-symbol captures whose primary required that retry.

## Pointers

- `ARCHITECTURE.md`: contract and divergence sections (baseline `3b56eef7`).
