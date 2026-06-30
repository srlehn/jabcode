package decode

import "image"

// pointF is a 2D point with floating-point coordinates. The stdlib image.Point
// is integer-only, so detection geometry uses this.
type pointF struct{ x, y float64 }

// perspective is a 3x3 projective transform matrix.
type perspective struct {
	a11, a12, a13 float64
	a21, a22, a23 float64
	a31, a32, a33 float64
}

// square2Quad returns the transform mapping the unit square to the quadrilateral
// with the given corners.
func square2Quad(x0, y0, x1, y1, x2, y2, x3, y3 float64) perspective {
	// Ports square2Quad in transform.c.
	dx3 := x0 - x1 + x2 - x3
	dy3 := y0 - y1 + y2 - y3
	if dx3 == 0 && dy3 == 0 {
		return perspective{
			a11: x1 - x0, a21: x2 - x1, a31: x0,
			a12: y1 - y0, a22: y2 - y1, a32: y0,
			a13: 0, a23: 0, a33: 1,
		}
	}
	dx1 := x1 - x2
	dx2 := x3 - x2
	dy1 := y1 - y2
	dy2 := y3 - y2
	denom := dx1*dy2 - dx2*dy1
	a13 := (dx3*dy2 - dx2*dy3) / denom
	a23 := (dx1*dy3 - dx3*dy1) / denom
	return perspective{
		a11: x1 - x0 + a13*x1, a21: x3 - x0 + a23*x3, a31: x0,
		a12: y1 - y0 + a13*y1, a22: y3 - y0 + a23*y3, a32: y0,
		a13: a13, a23: a23, a33: 1,
	}
}

// quad2Square returns the transform mapping the given quadrilateral to the unit
// square: the adjugate of square2Quad.
func quad2Square(x0, y0, x1, y1, x2, y2, x3, y3 float64) perspective {
	// Ports quad2Square in transform.c.
	s := square2Quad(x0, y0, x1, y1, x2, y2, x3, y3)
	return perspective{
		a11: s.a22*s.a33 - s.a23*s.a32, a21: s.a23*s.a31 - s.a21*s.a33, a31: s.a21*s.a32 - s.a22*s.a31,
		a12: s.a13*s.a32 - s.a12*s.a33, a22: s.a11*s.a33 - s.a13*s.a31, a32: s.a12*s.a31 - s.a11*s.a32,
		a13: s.a12*s.a23 - s.a13*s.a22, a23: s.a13*s.a21 - s.a11*s.a23, a33: s.a11*s.a22 - s.a12*s.a21,
	}
}

// mul returns m·n.
func (m perspective) mul(n perspective) perspective {
	// Ports multiply in transform.c.
	return perspective{
		a11: m.a11*n.a11 + m.a12*n.a21 + m.a13*n.a31,
		a21: m.a21*n.a11 + m.a22*n.a21 + m.a23*n.a31,
		a31: m.a31*n.a11 + m.a32*n.a21 + m.a33*n.a31,
		a12: m.a11*n.a12 + m.a12*n.a22 + m.a13*n.a32,
		a22: m.a21*n.a12 + m.a22*n.a22 + m.a23*n.a32,
		a32: m.a31*n.a12 + m.a32*n.a22 + m.a33*n.a32,
		a13: m.a11*n.a13 + m.a12*n.a23 + m.a13*n.a33,
		a23: m.a21*n.a13 + m.a22*n.a23 + m.a23*n.a33,
		a33: m.a31*n.a13 + m.a32*n.a23 + m.a33*n.a33,
	}
}

// quadToQuad returns the transform mapping source quadrilateral s to destination
// quadrilateral d.
func quadToQuad(s, d [4]pointF) perspective {
	// Ports perspectiveTransform in transform.c.
	q2s := quad2Square(s[0].x, s[0].y, s[1].x, s[1].y, s[2].x, s[2].y, s[3].x, s[3].y)
	s2q := square2Quad(d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y)
	return q2s.mul(s2q)
}

// getPerspectiveTransform returns the transform mapping a symbol's module grid
// (corners at 3.5 inside each finder/alignment pattern) to the four detected
// pattern centers.
func getPerspectiveTransform(p0, p1, p2, p3 pointF, side image.Point) perspective {
	// Ports getPerspectiveTransform in transform.c.
	sx, sy := float64(side.X), float64(side.Y)
	src := [4]pointF{{3.5, 3.5}, {sx - 3.5, 3.5}, {sx - 3.5, sy - 3.5}, {3.5, sy - 3.5}}
	dst := [4]pointF{p0, p1, p2, p3}
	return quadToQuad(src, dst)
}

// warp maps a point through the transform.
func (m perspective) warp(p pointF) pointF {
	// Ports warpPoints in transform.c.
	denom := m.a13*p.x + m.a23*p.y + m.a33
	return pointF{
		x: (m.a11*p.x + m.a21*p.y + m.a31) / denom,
		y: (m.a12*p.x + m.a22*p.y + m.a32) / denom,
	}
}
