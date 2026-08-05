//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

//go:embed shaders/alignment_search.wgsl
var alignmentSearchWGSL string

// Parameter word indices, matching alignment_search.wgsl.
const (
	gpuAlignParamWidth    = 0
	gpuAlignParamHeight   = 1
	gpuAlignParamNAPX     = 2
	gpuAlignParamNAPY     = 3
	gpuAlignParamDiagonal = 4
	gpuAlignParamCoreR    = 5
	gpuAlignParamCoreG    = 6
	gpuAlignParamCoreB    = 7
	gpuAlignParamSideX    = 8
	gpuAlignParamSideY    = 9
	gpuAlignParamModuleMx = 10
	gpuAlignParamQuad     = 11
	gpuAlignParamAPPosX   = 19
	gpuAlignParamAPPosY   = 28
	gpuAlignParamWords    = 37
)

// The per-cell record the kernel carries between diagonals and hands back.
const (
	gpuAlignCellWords  = 6
	gpuAlignCellFound  = 0
	gpuAlignCellCX     = 1
	gpuAlignCellCY     = 2
	gpuAlignCellModule = 3
	gpuAlignCellDir    = 4
)

// gpuAlignMaxCells bounds the alignment grid. APNum tops out at nine per axis,
// so one allocation covers every symbol version.
const gpuAlignMaxCells = 9 * 9

// initializeAlignment allocates the cell table and compiles the search. Like
// the sampler it runs with the rest of the resident stage set, so the compile
// lands in warm-up rather than on the decode call.
func (resident *gpuResidentBinarizer) initializeAlignment() error {
	var err error
	resident.alignCells, err = resident.device.NewBuffer(gpuAlignMaxCells * gpuAlignCellWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment cells: %w", err)
	}
	resident.alignParams, err = resident.device.NewBuffer(gpuAlignParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment parameters: %w", err)
	}
	resident.alignKernel, err = resident.kernels.alignmentSearch()
	if err != nil {
		return err
	}
	resident.alignBindings, err = resident.alignKernel.NewBindings(
		vulki.BindBuffer(0, resident.binarizer.packedMasks),
		vulki.BindBuffer(1, resident.alignCells),
		vulki.BindBuffer(2, resident.alignParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU alignment search: %w", err)
	}
	return nil
}

// alignmentGrid is one symbol's alignment-pattern search request: the grid
// shape, the finder quad that seeds its corners and axes, and the tables that
// place each cell in module coordinates.
type alignmentGrid struct {
	nApX, nApY int
	sideX      int
	sideY      int
	apType     int
	corners    [4]FinderPattern
	posX       []int
	posY       []int
}

// SearchAlignment locates every interior alignment pattern against the masks
// the pass left resident, and hands back the cell table.
//
// The whole grid costs one submission. A cell's prediction is extrapolated from
// the neighbours above and to its left, so the grid is walked one anti-diagonal
// at a time, but the diagonals are separated by barriers inside a single
// recording rather than by round trips: the host never sees an intermediate
// cell, and the masks it would otherwise have needed never leave the device.
func (resident *gpuResidentBinarizer) SearchAlignment(
	width, height int,
	grid alignmentGrid,
) ([]FinderPattern, error) {
	if resident == nil || resident.closed || resident.alignBindings == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	cells := grid.nApX * grid.nApY
	if grid.nApX <= 0 || grid.nApY <= 0 || cells > gpuAlignMaxCells {
		return nil, fmt.Errorf("jabcode: GPU alignment grid %dx%d is out of range", grid.nApX, grid.nApY)
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return nil, fmt.Errorf("jabcode: GPU alignment dimensions are unavailable")
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU alignment recorder: %w", err)
	}
	defer recorder.Abort()

	seed := gpuAlignmentSeed(grid)
	if err := recorder.Update(resident.alignCells, 0, seed); err != nil {
		return nil, fmt.Errorf("jabcode: seed GPU alignment cells: %w", err)
	}
	if err := recorder.Barrier(resident.binarizer.packedMasks, resident.alignCells); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU alignment inputs: %w", err)
	}
	for diagonal := range grid.nApX + grid.nApY - 1 {
		params := gpuAlignmentParams(width, height, grid, diagonal)
		if err := recorder.Update(resident.alignParams, 0, params[:]); err != nil {
			return nil, fmt.Errorf("jabcode: update GPU alignment parameters: %w", err)
		}
		if err := recorder.Barrier(resident.alignParams); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU alignment parameters: %w", err)
		}
		// One workgroup per cell on this diagonal; a diagonal never holds more
		// cells than the shorter grid axis.
		span := min(grid.nApX, grid.nApY)
		if err := recorder.Dispatch(
			resident.alignKernel,
			resident.alignBindings,
			vulki.Workgroups{X: uint32(span), Y: 1, Z: 1},
		); err != nil {
			return nil, fmt.Errorf("jabcode: dispatch GPU alignment diagonal %d: %w", diagonal, err)
		}
		if err := recorder.Barrier(resident.alignCells); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU alignment diagonal %d: %w", diagonal, err)
		}
	}
	table := make([]byte, cells*gpuAlignCellWords*4)
	phaseprobe.Count("download.alignment_cells", len(table))
	if err := recorder.Download(resident.alignCells, 0, table); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU alignment download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: run GPU alignment search: %w", err)
	}
	return parseAlignmentCells(table, cells, grid.apType), nil
}

// gpuAlignmentSeed lays down the four quad corners the search extrapolates
// from. They are measurements the finder stage already made, so the kernel
// treats them as found and never searches them.
func gpuAlignmentSeed(grid alignmentGrid) []byte {
	seed := make([]byte, grid.nApX*grid.nApY*gpuAlignCellWords*4)
	corners := [4]int{
		0,
		grid.nApX - 1,
		(grid.nApY-1)*grid.nApX + grid.nApX - 1,
		(grid.nApY - 1) * grid.nApX,
	}
	for at, index := range corners {
		fp := grid.corners[at]
		base := index * gpuAlignCellWords * 4
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellFound*4:], 1)
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellCX*4:], math.Float32bits(float32(fp.Center.X)))
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellCY*4:], math.Float32bits(float32(fp.Center.Y)))
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellModule*4:], math.Float32bits(float32(fp.ModuleSize)))
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellDir*4:], uint32(int32(fp.direction)))
	}
	return seed
}

func gpuAlignmentParams(width, height int, grid alignmentGrid, diagonal int) [gpuAlignParamWords * 4]byte {
	var params [gpuAlignParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	putF := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuAlignParamWidth, uint32(width))
	put(gpuAlignParamHeight, uint32(height))
	put(gpuAlignParamNAPX, uint32(grid.nApX))
	put(gpuAlignParamNAPY, uint32(grid.nApY))
	put(gpuAlignParamDiagonal, uint32(diagonal))
	// The mask bit is the binarized channel, so the core's palette component
	// reduces to whether that channel is on for this pattern type.
	core := apCoreColorIndex(grid.apType) * 3
	for channel := range 3 {
		bit := uint32(0)
		if palette.Default[core+channel] != 0 {
			bit = 1
		}
		put(gpuAlignParamCoreR+channel, bit)
	}
	put(gpuAlignParamSideX, uint32(grid.sideX))
	put(gpuAlignParamSideY, uint32(grid.sideY))
	// The ceiling the host's run test uses: a core run at or above it is a
	// field of colour rather than a pattern core.
	putF(gpuAlignParamModuleMx, gpuAlignmentModuleCeiling(grid))
	for at := range 4 {
		putF(gpuAlignParamQuad+at*2, grid.corners[at].Center.X)
		putF(gpuAlignParamQuad+at*2+1, grid.corners[at].Center.Y)
	}
	for at := range grid.posX {
		put(gpuAlignParamAPPosX+at, uint32(grid.posX[at]))
	}
	for at := range grid.posY {
		put(gpuAlignParamAPPosY+at, uint32(grid.posY[at]))
	}
	return params
}

// gpuAlignmentModuleCeiling derives the largest core run the search will accept
// from the quad's own scale, so nothing here is a fixed pixel bound: the
// corners span a known number of modules, and a core run may not exceed a small
// multiple of one of them.
func gpuAlignmentModuleCeiling(grid alignmentGrid) float64 {
	span := 0.0
	for _, fp := range grid.corners {
		span += fp.ModuleSize
	}
	return 3 * span / 4
}

func parseAlignmentCells(table []byte, cells, apType int) []FinderPattern {
	out := make([]FinderPattern, cells)
	for index := range out {
		base := index * gpuAlignCellWords * 4
		found := binary.LittleEndian.Uint32(table[base+gpuAlignCellFound*4:])
		out[index] = FinderPattern{
			Typ: apType,
			Center: core.PointF{
				X: float64(math.Float32frombits(binary.LittleEndian.Uint32(table[base+gpuAlignCellCX*4:]))),
				Y: float64(math.Float32frombits(binary.LittleEndian.Uint32(table[base+gpuAlignCellCY*4:]))),
			},
			ModuleSize: float64(math.Float32frombits(binary.LittleEndian.Uint32(table[base+gpuAlignCellModule*4:]))),
			direction:  int(int32(binary.LittleEndian.Uint32(table[base+gpuAlignCellDir*4:]))),
		}
		if found != 0 {
			out[index].FoundCount = 1
		}
	}
	return out
}
