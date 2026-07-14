// Package tables holds the JAB Code static lookup tables (encoding,
// alignment-pattern, and palette/finder geometry) shared by the encoder and
// decoder.
package tables

import (
	"image"

	"github.com/srlehn/jabcode/internal/wire"
)

// Code generated from the reference encoder.h; DO NOT EDIT.
// (jab_enconing_table, latch_shift_to, character_size, mode_switch)

// EncMax marks an impossible/unbounded mode transition.
const EncMax = 1000000 // ENC_MAX

// EncodingTable maps a byte value and encoding mode (upper, lower, numeric,
// punct, mixed, alphanumeric) to its code, or a negative sentinel.
var EncodingTable = [256][6]int{
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 16, -1},
	{-1, -1, -1, -1, 17, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -19, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{0, 0, 0, -1, -1, 0},
	{-1, -1, -1, 0, -1, -1},
	{-1, -1, -1, 1, -1, -1},
	{-1, -1, -1, -1, 0, -1},
	{-1, -1, -1, 2, -1, -1},
	{-1, -1, -1, 3, -1, -1},
	{-1, -1, -1, 4, -1, -1},
	{-1, -1, -1, 5, -1, -1},
	{-1, -1, -1, 6, -1, -1},
	{-1, -1, -1, 7, -1, -1},
	{-1, -1, -1, -1, 1, -1},
	{-1, -1, -1, -1, 2, -1},
	{-1, -1, 11, 8, -20, -1},
	{-1, -1, -1, 9, -1, -1},
	{-1, -1, 12, 10, -21, -1},
	{-1, -1, -1, 11, -1, -1},
	{-1, -1, 1, -1, -1, 1},
	{-1, -1, 2, -1, -1, 2},
	{-1, -1, 3, -1, -1, 3},
	{-1, -1, 4, -1, -1, 4},
	{-1, -1, 5, -1, -1, 5},
	{-1, -1, 6, -1, -1, 6},
	{-1, -1, 7, -1, -1, 7},
	{-1, -1, 8, -1, -1, 8},
	{-1, -1, 9, -1, -1, 9},
	{-1, -1, 10, -1, -1, 10},
	{-1, -1, -1, 12, -22, -1},
	{-1, -1, -1, 13, -1, -1},
	{-1, -1, -1, -1, 3, -1},
	{-1, -1, -1, -1, 4, -1},
	{-1, -1, -1, -1, 5, -1},
	{-1, -1, -1, 14, -1, -1},
	{-1, -1, -1, 15, -1, -1},
	{1, -1, -1, -1, -1, 11},
	{2, -1, -1, -1, -1, 12},
	{3, -1, -1, -1, -1, 13},
	{4, -1, -1, -1, -1, 14},
	{5, -1, -1, -1, -1, 15},
	{6, -1, -1, -1, -1, 16},
	{7, -1, -1, -1, -1, 17},
	{8, -1, -1, -1, -1, 18},
	{9, -1, -1, -1, -1, 19},
	{10, -1, -1, -1, -1, 20},
	{11, -1, -1, -1, -1, 21},
	{12, -1, -1, -1, -1, 22},
	{13, -1, -1, -1, -1, 23},
	{14, -1, -1, -1, -1, 24},
	{15, -1, -1, -1, -1, 25},
	{16, -1, -1, -1, -1, 26},
	{17, -1, -1, -1, -1, 27},
	{18, -1, -1, -1, -1, 28},
	{19, -1, -1, -1, -1, 29},
	{20, -1, -1, -1, -1, 30},
	{21, -1, -1, -1, -1, 31},
	{22, -1, -1, -1, -1, 32},
	{23, -1, -1, -1, -1, 33},
	{24, -1, -1, -1, -1, 34},
	{25, -1, -1, -1, -1, 35},
	{26, -1, -1, -1, -1, 36},
	{-1, -1, -1, -1, 6, -1},
	{-1, -1, -1, -1, 7, -1},
	{-1, -1, -1, -1, 8, -1},
	{-1, -1, -1, -1, 9, -1},
	{-1, -1, -1, -1, 10, -1},
	{-1, -1, -1, -1, 11, -1},
	{-1, 1, -1, -1, -1, 37},
	{-1, 2, -1, -1, -1, 38},
	{-1, 3, -1, -1, -1, 39},
	{-1, 4, -1, -1, -1, 40},
	{-1, 5, -1, -1, -1, 41},
	{-1, 6, -1, -1, -1, 42},
	{-1, 7, -1, -1, -1, 43},
	{-1, 8, -1, -1, -1, 44},
	{-1, 9, -1, -1, -1, 45},
	{-1, 10, -1, -1, -1, 46},
	{-1, 11, -1, -1, -1, 47},
	{-1, 12, -1, -1, -1, 48},
	{-1, 13, -1, -1, -1, 49},
	{-1, 14, -1, -1, -1, 50},
	{-1, 15, -1, -1, -1, 51},
	{-1, 16, -1, -1, -1, 52},
	{-1, 17, -1, -1, -1, 53},
	{-1, 18, -1, -1, -1, 54},
	{-1, 19, -1, -1, -1, 55},
	{-1, 20, -1, -1, -1, 56},
	{-1, 21, -1, -1, -1, 57},
	{-1, 22, -1, -1, -1, 58},
	{-1, 23, -1, -1, -1, 59},
	{-1, 24, -1, -1, -1, 60},
	{-1, 25, -1, -1, -1, 61},
	{-1, 26, -1, -1, -1, 62},
	{-1, -1, -1, -1, 12, -1},
	{-1, -1, -1, -1, 13, -1},
	{-1, -1, -1, -1, 14, -1},
	{-1, -1, -1, -1, 15, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 23, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 24, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 25, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 26, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 27, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 28, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 29, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 30, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 31, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1},
}

// LatchShiftTo[k][j] is the bit cost of switching from mode k to mode j
// (first latch, then shift); modes 0-6 are latch, 7-13 shift.
var LatchShiftTo = [14][14]int{
	{0, 5, 5, 1000000, 1000000, 5, 1000000, 1000000, 1000000, 1000000, 5, 7, 1000000, 11},
	{7, 0, 5, 1000000, 1000000, 5, 1000000, 5, 1000000, 1000000, 5, 7, 1000000, 11},
	{4, 6, 0, 1000000, 1000000, 9, 1000000, 6, 1000000, 1000000, 4, 6, 1000000, 10},
	{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0, 1000000, 1000000, 0, 1000000},
	{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0, 1000000, 1000000, 0, 1000000},
	{8, 13, 13, 1000000, 1000000, 0, 1000000, 1000000, 1000000, 1000000, 8, 8, 1000000, 12},
	{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0, 0, 1000000, 1000000, 0, 0},
	{0, 5, 5, 1000000, 1000000, 5, 1000000, 1000000, 1000000, 1000000, 5, 7, 1000000, 11},
	{7, 0, 5, 1000000, 1000000, 5, 1000000, 5, 1000000, 1000000, 5, 7, 1000000, 11},
	{4, 6, 0, 1000000, 1000000, 9, 1000000, 6, 1000000, 1000000, 4, 6, 1000000, 10},
	{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0, 1000000, 1000000, 0, 1000000},
	{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0, 1000000, 1000000, 0, 1000000},
	{8, 13, 13, 1000000, 1000000, 0, 1000000, 1000000, 1000000, 1000000, 8, 8, 1000000, 12},
	{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0, 0, 1000000, 1000000, 0, 0},
}

// CharacterSize is the per-character bit size of each base mode.
var CharacterSize = [7]int{5, 5, 4, 4, 5, 6, 8}

// ModeSwitch[from][to] is the bit code written for a mode switch; the last
// two columns are ECI and FNC1.
var ModeSwitch = [7][16]int{
	{-1, 28, 29, -1, -1, 30, -1, -1, -1, -1, 27, 125, -1, 124, 126, -1},
	{126, -1, 29, -1, -1, 30, -1, 28, -1, 127, 27, 125, -1, 124, -1, 127},
	{14, 63, -1, -1, -1, 478, -1, 62, -1, -1, 13, 61, -1, 60, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{255, 8188, 8189, -1, -1, -1, -1, -1, -1, -1, 254, 253, -1, 252, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
}

// APNum is the number of finder/alignment patterns per row/column for
// side-versions 1..32.
var APNum = [32]int{2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9}

// APPos holds the finder/alignment pattern coordinates for side-versions 1..32.
var APPos = [32][9]int{
	{4, 18, 0, 0, 0, 0, 0, 0, 0},
	{4, 22, 0, 0, 0, 0, 0, 0, 0},
	{4, 26, 0, 0, 0, 0, 0, 0, 0},
	{4, 30, 0, 0, 0, 0, 0, 0, 0},
	{4, 34, 0, 0, 0, 0, 0, 0, 0},
	{4, 17, 38, 0, 0, 0, 0, 0, 0},
	{4, 20, 42, 0, 0, 0, 0, 0, 0},
	{4, 23, 46, 0, 0, 0, 0, 0, 0},
	{4, 26, 50, 0, 0, 0, 0, 0, 0},
	{4, 14, 32, 54, 0, 0, 0, 0, 0},
	{4, 17, 39, 58, 0, 0, 0, 0, 0},
	{4, 20, 46, 62, 0, 0, 0, 0, 0},
	{4, 23, 44, 66, 0, 0, 0, 0, 0},
	{4, 26, 37, 51, 70, 0, 0, 0, 0},
	{4, 14, 36, 58, 74, 0, 0, 0, 0},
	{4, 17, 39, 56, 78, 0, 0, 0, 0},
	{4, 20, 42, 63, 82, 0, 0, 0, 0},
	{4, 23, 38, 54, 70, 86, 0, 0, 0},
	{4, 26, 38, 56, 77, 90, 0, 0, 0},
	{4, 14, 33, 53, 72, 94, 0, 0, 0},
	{4, 17, 38, 59, 79, 98, 0, 0, 0},
	{4, 20, 36, 53, 70, 86, 102, 0, 0},
	{4, 23, 36, 55, 74, 93, 106, 0, 0},
	{4, 26, 36, 58, 79, 100, 110, 0, 0},
	{4, 14, 36, 58, 80, 92, 114, 0, 0},
	{4, 17, 34, 52, 70, 88, 99, 118, 0},
	{4, 20, 37, 54, 72, 89, 106, 122, 0},
	{4, 23, 38, 56, 74, 92, 113, 126, 0},
	{4, 26, 36, 58, 78, 98, 120, 130, 0},
	{4, 14, 32, 49, 67, 84, 102, 112, 134},
	{4, 17, 35, 53, 71, 89, 107, 119, 138},
	{4, 20, 38, 55, 73, 91, 108, 126, 142},
}

// NcColorEncode maps each 3-bit metadata value to two module color indices.
var NcColorEncode = [8][2]int{
	{0, 0},
	{0, 3},
	{0, 6},
	{3, 0},
	{3, 3},
	{3, 6},
	{6, 0},
	{6, 3},
}

// NcMetadataColorIndex maps an NcColorEncode value - black (0), cyan (3) or
// yellow (6) in the 8-color palette - to the palette index carrying that color in
// color mode nc. Part I is read before the palette, by module color pattern
// alone, so it must be placed in colors DecodeModuleNC recognizes: black, cyan,
// yellow. In the 4- and 8-color palettes those sit at indices 0/3/6 and this is
// the identity, but the higher modes place them elsewhere on the RGB grid, where
// the fixed index would render as an unrelated color. The finder cyan (FP3) and
// yellow (FP2) core columns already hold those per-mode indices.
func NcMetadataColorIndex(value, nc int) int {
	return NcMetadataColorIndexProfile(value, nc, wire.ISO23634)
}

// NcMetadataColorIndexProfile is NcMetadataColorIndex under the selected
// wire-format profile.
func NcMetadataColorIndexProfile(value, nc int, profile wire.Profile) int {
	if profile.UsesISO23634Base() && nc == 1 {
		switch value {
		case 3: // cyan
			return 1
		case 6: // yellow
			return 3
		default: // black
			return 0
		}
	}
	switch value {
	case 3: // cyan
		return FPCoreColor[3][nc]
	case 6: // yellow
		return FPCoreColor[2][nc]
	default: // black (value 0)
		return 0
	}
}

// PrimaryPalettePlacement / SecondaryPalettePlacement give the module order of
// the embedded color palette. They cover only the eight indices of a 4- or
// 8-color symbol; PrimaryPalettePlacementIndex / SecondaryPalettePlacementIndex
// extend them for higher color counts.
var PrimaryPalettePlacement = [4][8]int{
	{0, 3, 5, 6, 1, 2, 4, 7},
	{0, 6, 5, 3, 1, 2, 4, 7},
	{6, 0, 5, 3, 1, 2, 4, 7},
	{3, 0, 5, 6, 1, 2, 4, 7},
}

var SecondaryPalettePlacement = [8]int{3, 6, 5, 0, 1, 2, 4, 7}

// PrimaryPalettePlacementIndex returns which palette color index copy c places at
// palette slot i. For the eight low indices it is the reference-defined
// shuffle (kept byte-identical for 4- and 8-color symbols); above 7 neither the
// reference nor ISO defines an order, so the extension is the identity - every
// copy carries slot i as color i. Encoder and decoder both route through this
// function, so higher-color palettes round-trip regardless of the choice.
func PrimaryPalettePlacementIndex(c, i int) int {
	return PrimaryPalettePlacementIndexProfile(c, i, 8, wire.ISO23634)
}

// PrimaryPalettePlacementIndexProfile is PrimaryPalettePlacementIndex under
// the selected wire-format profile.
func PrimaryPalettePlacementIndexProfile(c, i, colorNumber int, profile wire.Profile) int {
	if profile.UsesISO23634Base() && colorNumber == 4 && i < 4 {
		iso4 := [4][4]int{
			{0, 1, 2, 3},
			{0, 3, 2, 1},
			{3, 0, 2, 1},
			{1, 0, 2, 3},
		}
		return iso4[c][i]
	}
	if i < len(PrimaryPalettePlacement[c]) {
		return PrimaryPalettePlacement[c][i]
	}
	return i
}

// SecondaryPalettePlacementIndex is the secondary-symbol counterpart of
// PrimaryPalettePlacementIndex.
func SecondaryPalettePlacementIndex(i int) int {
	return SecondaryPalettePlacementIndexProfile(i, 8, wire.ISO23634)
}

// SecondaryPalettePlacementIndexProfile is SecondaryPalettePlacementIndex
// under the selected wire-format profile.
func SecondaryPalettePlacementIndexProfile(i, colorNumber int, profile wire.Profile) int {
	if profile.UsesISO23634Base() && colorNumber == 4 && i < 4 {
		return [4]int{1, 3, 2, 0}[i]
	}
	if i < len(SecondaryPalettePlacement) {
		return SecondaryPalettePlacement[i]
	}
	return i
}

// FPCoreColor[fp][Nc] and APNCoreColor/APXCoreColor[Nc] are the finder and
// alignment pattern core color indices per color mode Nc.
var FPCoreColor = [4][8]int{
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0},
	{0, 2, 6, 14, 30, 60, 124, 252},
	{0, 3, 3, 3, 7, 15, 15, 31},
}

var APNCoreColor = [8]int{0, 3, 3, 3, 7, 15, 15, 31}

var APXCoreColor = [8]int{0, 2, 6, 14, 30, 60, 124, 252}

// FPCoreColorIndex returns one finder-pattern core color index under the
// selected wire-format profile.
func FPCoreColorIndex(fp, nc int, profile wire.Profile) int {
	if profile.UsesISO23634Base() && nc == 1 {
		return [4]int{0, 0, 3, 1}[fp]
	}
	return FPCoreColor[fp][nc]
}

// APNCoreColorIndex returns the U/L alignment-pattern core color index under
// the selected wire-format profile.
func APNCoreColorIndex(nc int, profile wire.Profile) int {
	if profile.UsesISO23634Base() && nc == 1 {
		return 1
	}
	return APNCoreColor[nc]
}

// APXCoreColorIndex returns the X0/X1 alignment-pattern core color index under
// the selected wire-format profile.
func APXCoreColorIndex(nc int, profile wire.Profile) int {
	if profile.UsesISO23634Base() && nc == 1 {
		return 3
	}
	return APXCoreColor[nc]
}

// SymbolPos gives the grid position of each cascaded symbol slot.
var SymbolPos = [61]image.Point{
	{0, 0},
	{0, -1},
	{0, 1},
	{-1, 0},
	{1, 0},
	{0, -2},
	{-1, -1},
	{1, -1},
	{0, 2},
	{-1, 1},
	{1, 1},
	{-2, 0},
	{2, 0},
	{0, -3},
	{-1, -2},
	{1, -2},
	{-2, -1},
	{2, -1},
	{0, 3},
	{-1, 2},
	{1, 2},
	{-2, 1},
	{2, 1},
	{-3, 0},
	{3, 0},
	{0, -4},
	{-1, -3},
	{1, -3},
	{-2, -2},
	{2, -2},
	{-3, -1},
	{3, -1},
	{0, 4},
	{-1, 3},
	{1, 3},
	{-2, 2},
	{2, 2},
	{-3, 1},
	{3, 1},
	{-4, 0},
	{4, 0},
	{0, -5},
	{-1, -4},
	{1, -4},
	{-2, -3},
	{2, -3},
	{-3, -2},
	{3, -2},
	{-4, -1},
	{4, -1},
	{0, 5},
	{-1, 4},
	{1, 4},
	{-2, 3},
	{2, 3},
	{-3, 2},
	{3, 2},
	{-4, 1},
	{4, 1},
	{-5, 0},
	{5, 0},
}

// SecondaryPalettePosition gives the position of the first 32 palette modules
// in a secondary symbol.
var SecondaryPalettePosition = [32]image.Point{
	{4, 5},
	{4, 6},
	{4, 7},
	{4, 8},
	{4, 9},
	{4, 10},
	{4, 11},
	{4, 12},
	{5, 12},
	{5, 11},
	{5, 10},
	{5, 9},
	{5, 8},
	{5, 7},
	{5, 6},
	{5, 5},
	{6, 5},
	{6, 6},
	{6, 7},
	{6, 8},
	{6, 9},
	{6, 10},
	{6, 11},
	{6, 12},
	{7, 12},
	{7, 11},
	{7, 10},
	{7, 9},
	{7, 8},
	{7, 7},
	{7, 6},
	{7, 5},
}
