package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

func TestMergeDecodedFinderHypothesisRequiresMessageAgreement(t *testing.T) {
	first := currentFinderHypothesisResult{
		data: &Message{
			Variant:            wire.ISO23634,
			Data:               []byte("payload"),
			ReaderTransmission: []byte("]j1payload"),
		},
	}
	first.patterns[0].FoundCount = 3
	var winner currentFinderHypothesisResult
	if !mergeDecodedFinderHypothesis(&winner, first) {
		t.Fatal("first decoded hypothesis was rejected")
	}

	same := first
	same.data = cloneMessage(first.data)
	same.patterns[0].FoundCount = 9
	if !mergeDecodedFinderHypothesis(&winner, same) {
		t.Fatal("matching decoded hypothesis was treated as ambiguous")
	}
	if winner.patterns[0].FoundCount != 3 {
		t.Fatal("matching hypothesis displaced the first successful geometry")
	}

	conflict := first
	conflict.data = cloneMessage(first.data)
	conflict.data.Data = []byte("different")
	conflict.data.ReaderTransmission = []byte("]j1different")
	if mergeDecodedFinderHypothesis(&winner, conflict) {
		t.Fatal("conflicting decoded hypotheses were accepted")
	}
}
