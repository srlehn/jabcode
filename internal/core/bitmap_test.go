package core

import (
	"image"
	"testing"
)

func TestBitmapViewsAliasMaterializedPixels(t *testing.T) {
	rgba := NewBitmap(3, 2, 4)
	rgba.Pix[4] = 17
	nrgba := rgba.NRGBA()
	if nrgba == nil || nrgba.Stride != 12 || nrgba.Rect.Dx() != 3 || nrgba.Rect.Dy() != 2 {
		t.Fatal("NRGBA view has incorrect shape")
	}
	nrgba.Pix[4] = 29
	if rgba.Pix[4] != 29 {
		t.Fatal("NRGBA view did not alias Bitmap pixels")
	}

	gray := NewBitmap(3, 2, 1)
	gray.Pix[4] = 31
	view := gray.Gray()
	if view == nil || view.Stride != 3 || view.Rect.Dx() != 3 || view.Rect.Dy() != 2 {
		t.Fatal("Gray view has incorrect shape")
	}
	view.Pix[4] = 43
	if gray.Pix[4] != 43 {
		t.Fatal("Gray view did not alias Bitmap pixels")
	}
}

func TestBitmapViewsRejectDeferredOrWrongPlanes(t *testing.T) {
	if NewBitmap(2, 2, 1).NRGBA() != nil {
		t.Fatal("NRGBA view accepted a one-channel bitmap")
	}
	if NewBitmap(2, 2, 4).Gray() != nil {
		t.Fatal("Gray view accepted a four-channel bitmap")
	}
	deferred := &Bitmap{Width: 2, Height: 2, Channels: 1}
	deferred.SetPixelReader(func(int, int) byte { return 255 })
	if deferred.Gray() != nil {
		t.Fatal("Gray view accepted a deferred bitmap")
	}
}

func BenchmarkBitmapFromImage(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = BitmapFromImage(img)
	}
}
