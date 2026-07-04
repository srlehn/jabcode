package detect

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/spec"
)

// findSecondarySymbol locates a secondary symbol docked to the given side of a
// host symbol by detecting its four corner alignment patterns.
func findSecondarySymbol(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int) bool {
	// Ports findSlaveSymbol in detector.c.
	var aps [4]FinderPattern

	secondary.SideSize = image.Pt(spec.VersionToSize(secondary.Meta.SideVersion.X), spec.VersionToSize(secondary.Meta.SideVersion.Y))

	hp := host.PatternPositions
	distx01, disty01 := hp[1].X-hp[0].X, hp[1].Y-hp[0].Y
	distx32, disty32 := hp[2].X-hp[3].X, hp[2].Y-hp[3].Y
	distx03, disty03 := hp[3].X-hp[0].X, hp[3].Y-hp[0].Y
	distx12, disty12 := hp[2].X-hp[1].X, hp[2].Y-hp[1].Y

	var alpha1, alpha2 float64
	sign := 1
	var dockedSideSize, undockedSideSize int
	var t1, t2, t3, t4, h1, h2 int

	switch dockedPosition {
	case 3:
		alpha1, alpha2, sign = math.Atan2(disty01, distx01), math.Atan2(disty32, distx32), 1
		dockedSideSize, undockedSideSize = secondary.SideSize.Y, secondary.SideSize.X
		t1, t2, t3, t4, h1, h2 = ap0, ap3, ap1, ap2, fp1, fp2
		secondary.HostPosition = 2
	case 2:
		alpha1, alpha2, sign = math.Atan2(disty32, distx32), math.Atan2(disty01, distx01), -1
		dockedSideSize, undockedSideSize = secondary.SideSize.Y, secondary.SideSize.X
		t1, t2, t3, t4, h1, h2 = ap2, ap1, ap3, ap0, fp3, fp0
		secondary.HostPosition = 3
	case 1:
		alpha1, alpha2, sign = math.Atan2(disty12, distx12), math.Atan2(disty03, distx03), 1
		dockedSideSize, undockedSideSize = secondary.SideSize.X, secondary.SideSize.Y
		t1, t2, t3, t4, h1, h2 = ap1, ap0, ap2, ap3, fp2, fp3
		secondary.HostPosition = 0
	case 0:
		alpha1, alpha2, sign = math.Atan2(disty03, distx03), math.Atan2(disty12, distx12), -1
		dockedSideSize, undockedSideSize = secondary.SideSize.X, secondary.SideSize.Y
		t1, t2, t3, t4, h1, h2 = ap3, ap2, ap0, ap1, fp0, fp1
		secondary.HostPosition = 1
	}
	signf := float64(sign)

	aps[t1].Center.X = hp[h1].X + signf*7*host.ModuleSize*math.Cos(alpha1)
	aps[t1].Center.Y = hp[h1].Y + signf*7*host.ModuleSize*math.Sin(alpha1)
	aps[t1] = findAlignmentPattern(ch, aps[t1].Center.X, aps[t1].Center.Y, host.ModuleSize, t1)
	if aps[t1].FoundCount == 0 {
		return false
	}
	aps[t2].Center.X = hp[h2].X + signf*7*host.ModuleSize*math.Cos(alpha2)
	aps[t2].Center.Y = hp[h2].Y + signf*7*host.ModuleSize*math.Sin(alpha2)
	aps[t2] = findAlignmentPattern(ch, aps[t2].Center.X, aps[t2].Center.Y, host.ModuleSize, t2)
	if aps[t2].FoundCount == 0 {
		return false
	}

	secondary.ModuleSize = math.Hypot(aps[t1].Center.X-aps[t2].Center.X, aps[t1].Center.Y-aps[t2].Center.Y) / float64(dockedSideSize-7)

	aps[t3].Center.X = aps[t1].Center.X + signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Cos(alpha1)
	aps[t3].Center.Y = aps[t1].Center.Y + signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Sin(alpha1)
	aps[t3] = findAlignmentPattern(ch, aps[t3].Center.X, aps[t3].Center.Y, secondary.ModuleSize, t3)
	aps[t4].Center.X = aps[t2].Center.X + signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Cos(alpha2)
	aps[t4].Center.Y = aps[t2].Center.Y + signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Sin(alpha2)
	aps[t4] = findAlignmentPattern(ch, aps[t4].Center.X, aps[t4].Center.Y, secondary.ModuleSize, t4)

	if aps[t3].FoundCount == 0 && aps[t4].FoundCount == 0 {
		return false
	}
	if aps[t3].FoundCount == 0 {
		avg24 := (aps[t2].ModuleSize + aps[t4].ModuleSize) / 2.0
		avg14 := (aps[t1].ModuleSize + aps[t4].ModuleSize) / 2.0
		aps[t3].Center.X = (aps[t4].Center.X-aps[t2].Center.X)/avg24*avg14 + aps[t1].Center.X
		aps[t3].Center.Y = (aps[t4].Center.Y-aps[t2].Center.Y)/avg24*avg14 + aps[t1].Center.Y
		aps[t3].Typ, aps[t3].FoundCount = t3, 1
		aps[t3].ModuleSize = (aps[t1].ModuleSize + aps[t2].ModuleSize + aps[t4].ModuleSize) / 3.0
		if aps[t3].Center.X > float64(bm.Width-1) || aps[t3].Center.Y > float64(bm.Height-1) {
			return false
		}
	}
	if aps[t4].FoundCount == 0 {
		avg13 := (aps[t1].ModuleSize + aps[t3].ModuleSize) / 2.0
		avg23 := (aps[t2].ModuleSize + aps[t3].ModuleSize) / 2.0
		aps[t4].Center.X = (aps[t3].Center.X-aps[t1].Center.X)/avg13*avg23 + aps[t2].Center.X
		aps[t4].Center.Y = (aps[t3].Center.Y-aps[t1].Center.Y)/avg13*avg23 + aps[t2].Center.Y
		aps[t4].Typ, aps[t4].FoundCount = t4, 1
		// Note: the reference averages ap1 twice here; kept identical.
		aps[t4].ModuleSize = (aps[t1].ModuleSize + aps[t1].ModuleSize + aps[t3].ModuleSize) / 3.0
		if aps[t4].Center.X > float64(bm.Width-1) || aps[t4].Center.Y > float64(bm.Height-1) {
			return false
		}
	}

	secondary.PatternPositions[t1] = aps[t1].Center
	secondary.PatternPositions[t2] = aps[t2].Center
	secondary.PatternPositions[t3] = aps[t3].Center
	secondary.PatternPositions[t4] = aps[t4].Center
	secondary.ModuleSize = (aps[t1].ModuleSize + aps[t2].ModuleSize + aps[t3].ModuleSize + aps[t4].ModuleSize) / 4.0
	return true
}

// DetectSecondary finds and samples a secondary symbol docked at the given
// position of a host symbol.
func DetectSecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int) *core.Bitmap {
	// Ports detectSlave in detector.c.
	if dockedPosition < 0 || dockedPosition > 3 {
		return nil
	}
	if !findSecondarySymbol(bm, ch, host, secondary, dockedPosition) {
		return nil
	}
	pt := core.PerspectiveTransform(secondary.PatternPositions[0], secondary.PatternPositions[1],
		secondary.PatternPositions[2], secondary.PatternPositions[3], secondary.SideSize)
	return SampleSymbol(bm, pt, secondary.SideSize)
}
