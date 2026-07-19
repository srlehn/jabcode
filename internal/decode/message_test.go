package decode

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

func TestDecodeMessageVariantKeepsRawDataAndECISeparate(t *testing.T) {
	var bits messageBits
	bits.upper(1)
	bits.byteRun('\\')
	bits.upper(2)
	bits.eci(26, 8)
	bits.byteRun('\\')
	bits.upper(3)

	message, ok := DecodeMessageVariant(bits, wire.ISO23634)
	if !ok {
		t.Fatal("DecodeMessageVariant rejected a valid ISO message")
	}
	if want := []byte{'A', '\\', 'B', '\\', 'C'}; !bytes.Equal(message.Data, want) {
		t.Fatalf("Data = %q, want %q", message.Data, want)
	}
	wantTransmission := []byte{']', 'j', '1', 'A', '\\', '\\', 'B', '\\', '0', '0', '0', '0', '2', '6', '\\', '\\', 'C'}
	if !bytes.Equal(message.ReaderTransmission, wantTransmission) {
		t.Fatalf("ReaderTransmission = %q, want %q", message.ReaderTransmission, wantTransmission)
	}
	wantControls := []MessageControl{{Kind: MessageControlECI, Offset: 3, Assignment: 26}}
	if !reflect.DeepEqual(message.Controls, wantControls) {
		t.Fatalf("Controls = %+v, want %+v", message.Controls, wantControls)
	}
}

func TestDecodeMessageVariantRecordsFNC1Structure(t *testing.T) {
	var bits messageBits
	bits.additional(4)
	bits.upper(1)
	bits.additional(4)
	bits.upper(2)
	bits.additional(5)

	message, ok := DecodeMessageVariant(bits, wire.ISO23634)
	if !ok {
		t.Fatal("DecodeMessageVariant rejected a valid FNC1 message")
	}
	if want := []byte{'A', 29, 'B'}; !bytes.Equal(message.Data, want) {
		t.Fatalf("Data = %q, want %q", message.Data, want)
	}
	wantControls := []MessageControl{
		{Kind: MessageControlFNC1Start, Offset: 0},
		{Kind: MessageControlFNC1Separator, Offset: 1},
		{Kind: MessageControlFNC1End, Offset: 3},
	}
	if !reflect.DeepEqual(message.Controls, wantControls) {
		t.Fatalf("Controls = %+v, want %+v", message.Controls, wantControls)
	}
}

func TestDecodeMessageVariantDoesNotParseIdentifierLookingData(t *testing.T) {
	payload := []byte("]j1\\000026")
	var bits messageBits
	bits.byteRun(payload...)

	message, ok := DecodeMessageVariant(bits, wire.ISO23634)
	if !ok {
		t.Fatal("DecodeMessageVariant rejected byte-mode data")
	}
	if !bytes.Equal(message.Data, payload) {
		t.Fatalf("Data = %q, want exact payload %q", message.Data, payload)
	}
	if len(message.Controls) != 0 {
		t.Fatalf("literal escape-looking data produced controls: %+v", message.Controls)
	}
}
