package decode

import (
	"bytes"

	"github.com/srlehn/jabcode/internal/core"
)

const maxModuleEvidenceEntries = 2

// ModuleEvidenceCache retains neutral payload-module classifications for one
// sampled primary matrix. It is deliberately fixed-size and map-free: the
// current physical family has at most two irreducible wire interpretations.
// Two entries retain the finder-grid sample and its first alignment fallback;
// a later unique fallback belongs to the final interpretation and has no
// subsequent consumer. Deterministic priority remains the caller's
// responsibility.
type ModuleEvidenceCache struct {
	entries [maxModuleEvidenceEntries]moduleEvidence
	count   int
}

type moduleEvidence struct {
	matrix        *core.Bitmap
	symbolType    int
	nc            int
	dataMap       []byte
	palette       []byte
	rawModules    []byte
	reliabilities []float64
}

func (cache *ModuleEvidenceCache) find(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, symbolType int) *moduleEvidence {
	if cache == nil {
		return nil
	}
	for i := range cache.count {
		entry := &cache.entries[i]
		if entry.matrix == matrix && entry.symbolType == symbolType && entry.nc == symbol.Meta.NC &&
			bytes.Equal(entry.dataMap, dataMap) && bytes.Equal(entry.palette, symbol.Palette) {
			return entry
		}
	}
	return nil
}

func (cache *ModuleEvidenceCache) add(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, symbolType int, rawModules []byte) *moduleEvidence {
	if cache == nil || cache.count >= len(cache.entries) {
		return nil
	}
	entry := &cache.entries[cache.count]
	cache.count++
	*entry = moduleEvidence{
		matrix:     matrix,
		symbolType: symbolType,
		nc:         symbol.Meta.NC,
		dataMap:    dataMap,
		palette:    symbol.Palette,
		rawModules: rawModules,
	}
	return entry
}
