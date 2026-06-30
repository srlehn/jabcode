package decode

import "image"

// sampleSymbol samples a side.X by side.Y matrix of module colors from the image
// using the perspective transform, taking a 3x3 neighborhood average for each
// module center. It returns an RGBA bitmap of the sampled module values, or nil
// if a module maps too far outside the image.
func sampleSymbol(bm *bitmap, pt perspective, side image.Point) *bitmap {
	// Ports sampleSymbol in sample.c.
	out := newBitmap(side.X, side.Y, bm.channels)
	bpp := bm.channels
	bytesPerRow := bm.width * bpp

	points := make([]pointF, side.X)
	for i := 0; i < side.Y; i++ {
		for j := 0; j < side.X; j++ {
			points[j] = pt.warp(pointF{float64(j) + 0.5, float64(i) + 0.5})
		}
		for j := 0; j < side.X; j++ {
			mx := int(points[j].x)
			my := int(points[j].y)
			if mx < 0 || mx > bm.width-1 {
				switch mx {
				case -1:
					mx = 0
				case bm.width:
					mx = bm.width - 1
				default:
					return nil
				}
			}
			if my < 0 || my > bm.height-1 {
				switch my {
				case -1:
					my = 0
				case bm.height:
					my = bm.height - 1
				default:
					return nil
				}
			}
			for c := 0; c < out.channels; c++ {
				sum := 0.0
				for dx := -1; dx <= 1; dx++ {
					for dy := -1; dy <= 1; dy++ {
						px, py := mx+dx, my+dy
						if px < 0 || px > bm.width-1 {
							px = mx
						}
						if py < 0 || py > bm.height-1 {
							py = my
						}
						sum += float64(bm.pix[py*bytesPerRow+px*bpp+c])
					}
				}
				out.pix[(i*side.X+j)*out.channels+c] = byte(sum/9.0 + 0.5)
			}
		}
	}
	return out
}
