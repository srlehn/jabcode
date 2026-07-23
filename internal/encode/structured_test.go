package encode

import (
	"testing"

	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestStructuredDataWireDecode(t *testing.T) {
	bits, err := encodeStructuredData([]byte("structured ECI"), []Control{{Kind: ControlECI, Assignment: 26}})
	if err != nil {
		t.Fatal(err)
	}
	message, ok := decode.DecodeMessageVariant(bits, wire.ISO23634)
	if !ok {
		t.Fatalf("wire decode rejected %d bits", len(bits))
	}
	if string(message.Data) != "structured ECI" {
		t.Fatalf("data = %q", message.Data)
	}
}
