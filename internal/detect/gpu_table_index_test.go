//go:build !js

package detect

import (
	"encoding/binary"
	"testing"

	"github.com/srlehn/vulki"
)

// gpuTableIndexWGSL writes the same four-entry table through each form a shader
// could hold it in, so the comparison is between the forms rather than between
// devices.
const gpuTableIndexWGSL = `
const CONST_TABLE: array<u32, 4> = array<u32, 4>(7u, 8u, 9u, 10u);

fn switched(index: u32) -> u32 {
    switch index {
        case 0u: { return 7u; }
        case 1u: { return 8u; }
        case 2u: { return 9u; }
        default: { return 10u; }
    }
}

@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    out[lane] = switched(lane);
    var assigned: array<u32, 4>;
    assigned[0] = 7u;
    assigned[1] = 8u;
    assigned[2] = 9u;
    assigned[3] = 10u;
    out[4u + lane] = assigned[lane];
    out[8u + lane] = CONST_TABLE[2u];
}
`

// TestGPUDynamicTableIndex pins the shader forms a runtime index may be applied
// to. It exists because the form the shaders used to prefer does not work:
// indexing a const array with a runtime value compiles to a silent zero on this
// toolchain, which walked every finder edge from pattern zero to pattern zero
// and made every device side estimate invalid. Nothing about that reports an
// error, so the only defence is a gate that reads the values back.
//
// The forms asserted here are the ones the shaders now use: a switch over the
// index, and an array assigned element by element before it is indexed. A
// constant index into a const array is fine and is checked alongside them.
func TestGPUDynamicTableIndex(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	defer func() {
		if err := device.Close(); err != nil {
			t.Errorf("close table index device: %v", err)
		}
	}()
	kernel, err := device.NewKernel(vulki.KernelOptions{
		WGSL:       gpuTableIndexWGSL,
		EntryPoint: "main",
		Bindings:   []vulki.BindingLayout{{Binding: 0, Access: vulki.BufferReadWrite}},
	})
	if err != nil {
		t.Fatalf("compile table index kernel: %v", err)
	}
	buffer, err := device.NewBuffer(12 * 4)
	if err != nil {
		t.Fatalf("allocate table index buffer: %v", err)
	}
	bindings, err := kernel.NewBindings(vulki.BindBuffer(0, buffer))
	if err != nil {
		t.Fatalf("bind table index kernel: %v", err)
	}
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("create table index recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		t.Fatalf("dispatch table index kernel: %v", err)
	}
	out := make([]byte, 12*4)
	if err := recorder.Download(buffer, 0, out); err != nil {
		t.Fatalf("record table index download: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run table index kernel: %v", err)
	}
	want := [12]uint32{7, 8, 9, 10, 7, 8, 9, 10, 9, 9, 9, 9}
	for at, value := range want {
		if got := binary.LittleEndian.Uint32(out[at*4:]); got != value {
			t.Fatalf("table word %d = %d, want %d", at, got, value)
		}
	}
}
