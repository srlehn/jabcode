//go:build !js

package detect

import (
	"image"
	"testing"

	"github.com/srlehn/vulki"
)

// TestGPULocalModuleCountsMatchHost holds the device edge walk to the host
// walk. The walk is serial and self-correcting, so an isolated disagreement
// about one window would normally be absorbed by the next step; what the test
// pins is the count each edge ends on, which is the only thing the side size
// is decided from.
//
// The cases sweep module size because that is what sets the walk's step length
// and the number of candidate offsets each step chooses between, which is where
// the device's parallel evaluation could diverge from the host's serial scan.
func TestGPULocalModuleCountsMatchHost(t *testing.T) {
	// The frame has to hold the widest case's whole symbol: a module size
	// large enough to push the candidate offsets past one workgroup spans well
	// over a thousand pixels across eleven modules.
	const width = 1601
	const height = 1201
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU module count test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU module count test input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU module count test device: %v", err)
		}
	})

	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU module count test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}
	balanced, err := resident.DownloadBalanced(width, height)
	if err != nil {
		t.Fatalf("download resident GPU balanced image: %v", err)
	}

	tests := []struct {
		name   string
		side   image.Point
		module float64
	}{
		{name: "small modules", side: image.Pt(41, 31), module: 4},
		{name: "medium modules", side: image.Pt(21, 17), module: 9},
		// A quarter of this module size exceeds the workgroup, so the walk
		// folds its candidates in chunks rather than in one pass.
		{name: "candidates past one workgroup", side: image.Pt(11, 9), module: 130},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			quad := gpuSampleTestQuad(width, height, test.side, test.module)
			fps := make([]FinderPattern, 4)
			for i := range fps {
				fps[i] = FinderPattern{
					Typ: i, Center: quad[i], ModuleSize: test.module, FoundCount: 1,
				}
			}
			want := LocalModuleCounts(balanced, fps)
			got, err := resident.LocalModuleCounts(width, height, fps)
			if err != nil {
				t.Fatalf("GPU LocalModuleCounts: %v", err)
			}
			if got != want {
				t.Errorf("GPU edge walk = %v, host = %v", got, want)
			}
			// A walk that declines on every edge would match trivially, so the
			// case has to prove it walked something.
			counted := 0
			for _, count := range want {
				if count > 0 {
					counted++
				}
			}
			if counted == 0 {
				t.Fatal("no edge produced a count, so the case proves nothing")
			}
			t.Logf("%s: counts %v over %d walked edges", test.name, got, counted)
		})
	}
}

// TestGPULocalModuleCountsRejectDegenerateEdges pins the walk's refusals: a
// zero-length edge and a missing module size have no trustworthy count, and the
// device has to decline them the same way the host does rather than emit a
// plausible number the side size would then trust.
func TestGPULocalModuleCountsRejectDegenerateEdges(t *testing.T) {
	const width = 129
	const height = 97
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU module count test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
	})
	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU module count test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}
	balanced, err := resident.DownloadBalanced(width, height)
	if err != nil {
		t.Fatalf("download resident GPU balanced image: %v", err)
	}

	side := image.Pt(17, 13)
	quad := gpuSampleTestQuad(width, height, side, 5)
	fps := make([]FinderPattern, 4)
	for i := range fps {
		fps[i] = FinderPattern{Typ: i, Center: quad[i], ModuleSize: 5, FoundCount: 1}
	}
	// SideEdges pairs {0,1} and {0,3} both start at finder 0, so collapsing it
	// onto finder 1 degenerates the first edge and clearing finder 3's module
	// size degenerates the third, leaving the other two to still count.
	fps[0].Center = fps[1].Center
	fps[3].ModuleSize = 0
	want := LocalModuleCounts(balanced, fps)
	if want[0] != -1 || want[2] != -1 {
		t.Fatalf("host walk did not decline the degenerate edges: %v", want)
	}
	got, err := resident.LocalModuleCounts(width, height, fps)
	if err != nil {
		t.Fatalf("GPU LocalModuleCounts: %v", err)
	}
	if got != want {
		t.Errorf("GPU edge walk = %v, host = %v", got, want)
	}
}
