//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/phaseprobe"
)

//go:embed shaders/local_module_count.wgsl
var localModuleCountWGSL string

// Parameter word indices, matching local_module_count.wgsl.
const (
	gpuModuleCountParamWidth  = 0
	gpuModuleCountParamHeight = 1
	gpuModuleCountParamEdges  = 2
	gpuModuleCountEdgeStride  = 6

	gpuModuleCountEdges      = 4
	gpuModuleCountParamWords = gpuModuleCountParamEdges +
		gpuModuleCountEdges*gpuModuleCountEdgeStride
	gpuModuleCountResultWords = gpuModuleCountEdges
)

// The device walks exactly the edges the host weighs, so the two counts must
// stay equal; this stops compiling if an edge is ever added or removed.
var _ [len(SideEdges) - gpuModuleCountEdges]struct{}

// initializeModuleCount allocates the edge walk's buffers and compiles it,
// alongside the rest of the resident stages so the compile lands in warm-up.
func (resident *gpuResidentBinarizer) initializeModuleCount() error {
	var err error
	resident.moduleCountResult, err = resident.device.NewBuffer(gpuModuleCountResultWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU module counts: %w", err)
	}
	resident.moduleCountParams, err = resident.device.NewBuffer(gpuModuleCountParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU module count parameters: %w", err)
	}
	resident.moduleCountKernel, err = resident.kernels.localModuleCount()
	if err != nil {
		return err
	}
	resident.moduleCountBindings, err = resident.moduleCountKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, resident.moduleCountResult),
		vulki.BindBuffer(2, resident.moduleCountParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU module count: %w", err)
	}
	return nil
}

// LocalModuleCounts walks all four finder-to-finder edges on the device, one
// workgroup each, and returns their module counts in SideEdges order. The walk
// reads a few hundred small windows of the balanced image, which is the only
// reason the host used to need the whole frame.
func (resident *gpuResidentBinarizer) LocalModuleCounts(
	width, height int,
	fps []FinderPattern,
) ([4]int, error) {
	var counts [4]int
	if resident == nil || resident.closed || resident.moduleCountBindings == nil {
		return counts, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if len(fps) < 4 {
		return counts, fmt.Errorf("jabcode: GPU module count needs four finders")
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return counts, fmt.Errorf("jabcode: GPU module count dimensions are unavailable")
	}

	params := gpuModuleCountParams(width, height, fps)
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return counts, fmt.Errorf("jabcode: create GPU module count recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recordGPUUpdate(
		recorder, "upload.module_count_params", resident.moduleCountParams, 0, params[:],
	); err != nil {
		return counts, fmt.Errorf("jabcode: update GPU module count parameters: %w", err)
	}
	if err := recorder.Dispatch(
		resident.moduleCountKernel,
		resident.moduleCountBindings,
		vulki.Workgroups{X: gpuModuleCountEdges, Y: 1, Z: 1},
	); err != nil {
		return counts, fmt.Errorf("jabcode: dispatch GPU module count: %w", err)
	}
	if err := recorder.Barrier(resident.moduleCountResult); err != nil {
		return counts, fmt.Errorf("jabcode: synchronize GPU module counts: %w", err)
	}
	result := make([]byte, gpuModuleCountResultWords*4)
	phaseprobe.Count("download.module_counts", len(result))
	if err := recorder.Download(resident.moduleCountResult, 0, result); err != nil {
		return counts, fmt.Errorf("jabcode: download GPU module counts: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return counts, fmt.Errorf("jabcode: run GPU module count: %w", err)
	}
	for i := range counts {
		counts[i] = int(int32(binary.LittleEndian.Uint32(result[i*4:])))
	}
	return counts, nil
}

func gpuModuleCountParams(width, height int, fps []FinderPattern) [gpuModuleCountParamWords * 4]byte {
	var params [gpuModuleCountParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	putFloat := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuModuleCountParamWidth, uint32(width))
	put(gpuModuleCountParamHeight, uint32(height))
	for i, edge := range SideEdges {
		base := gpuModuleCountParamEdges + i*gpuModuleCountEdgeStride
		a, b := fps[edge[0]], fps[edge[1]]
		putFloat(base+0, a.Center.X)
		putFloat(base+1, a.Center.Y)
		putFloat(base+2, a.ModuleSize)
		putFloat(base+3, b.Center.X)
		putFloat(base+4, b.Center.Y)
		putFloat(base+5, b.ModuleSize)
	}
	return params
}
