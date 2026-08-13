//go:build !js

package detect

import (
	"bytes"
	"encoding/binary"
	"image"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/wire"
)

// TestGPUMetadataParamsKernelMatchesHost runs metadata_params.wgsl on the
// device and compares the block it publishes against the host builder it
// replaced. The existing static-table test rebuilds the block in Go and never
// dispatches, so it cannot see a kernel that reads the wrong control words.
func TestGPUMetadataParamsKernelMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		if err := kernels.Close(); err != nil {
			t.Errorf("close GPU kernel set: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU device: %v", err)
		}
	})

	kernel, err := kernels.metadataParams()
	if err != nil {
		t.Fatalf("compile metadata parameter kernel: %v", err)
	}
	static := gpuMetadataStaticData()

	for _, variant := range []wire.Variant{wire.ISO23634, wire.ISOHighColor, wire.CurrentC} {
		for _, side := range []image.Point{image.Pt(21, 21), image.Pt(81, 85), image.Pt(145, 145)} {
			sampleParams := make([]byte, gpuSampleParamWords*4)
			binary.LittleEndian.PutUint32(
				sampleParams[gpuSampleParamDestWidth*4:], uint32(side.X))
			binary.LittleEndian.PutUint32(
				sampleParams[gpuSampleParamDestHeight*4:], uint32(side.Y))
			binary.LittleEndian.PutUint32(
				sampleParams[gpuSampleParamMetadataProfile*4:], gpuMetadataProfile(variant))

			var owned []*vulki.Buffer
			newBuffer := func(name string, data []byte) *vulki.Buffer {
				t.Helper()
				buffer, err := device.NewBuffer(uint64(len(data)))
				if err != nil {
					t.Fatalf("allocate %s: %v", name, err)
				}
				owned = append(owned, buffer)
				if err := buffer.Upload(data); err != nil {
					t.Fatalf("upload %s: %v", name, err)
				}
				return buffer
			}
			sample := newBuffer("sample parameters", sampleParams)
			grid := newBuffer("sampled grid", make([]byte, 4*4))
			params := newBuffer("metadata parameters", make([]byte, gpuMetadataParamWords*4))
			tables := newBuffer("static tables", static)

			bindings, err := kernel.NewBindings(
				vulki.BindBuffer(0, sample),
				vulki.BindBuffer(1, grid),
				vulki.BindBuffer(2, params),
				vulki.BindBuffer(3, tables),
			)
			if err != nil {
				t.Fatalf("bind metadata parameter kernel: %v", err)
			}

			recorder, err := device.NewRecorder()
			if err != nil {
				t.Fatalf("create recorder: %v", err)
			}
			got := make([]byte, gpuMetadataParamWords*4)
			if err := recorder.Dispatch(
				kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
			); err != nil {
				t.Fatalf("dispatch metadata parameter kernel: %v", err)
			}
			if err := recorder.Barrier(params); err != nil {
				t.Fatalf("synchronize metadata parameters: %v", err)
			}
			if err := recorder.Download(params, 0, got); err != nil {
				t.Fatalf("record metadata parameter download: %v", err)
			}
			if err := recorder.SubmitAndWait(); err != nil {
				t.Fatalf("run metadata parameter kernel: %v", err)
			}

			want := gpuMetadataParams(side, variant)
			if !bytes.Equal(got, want[:]) {
				first := -1
				for word := range gpuMetadataParamWords {
					g := binary.LittleEndian.Uint32(got[word*4:])
					w := binary.LittleEndian.Uint32(want[word*4:])
					if g != w {
						first = word
						t.Errorf("variant %d side %v: word %d = %d, want %d",
							variant, side, word, g, w)
						break
					}
				}
				t.Errorf("variant %d side %v: device block differs from host, first word %d",
					variant, side, first)
			}

			if err := bindings.Close(); err != nil {
				t.Errorf("close bindings: %v", err)
			}
			for i := len(owned) - 1; i >= 0; i-- {
				if err := owned[i].Close(); err != nil {
					t.Errorf("close buffer: %v", err)
				}
			}
		}
	}
}
