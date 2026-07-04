// Package core holds the types shared by the detection and decoding stages:
// the pixel Bitmap, floating-point geometry (PointF, Perspective), the
// decoded-symbol result types, the shared status codes, and small per-pixel
// colour statistics. It is the dependency leaf under detect, decode, read
// and diag, and imports none of them.
package core
