package read

import (
	"bytes"

	"github.com/srlehn/jabcode/internal/decode"
)

// Message is the mode decoder's paired raw-data and reader-transmission result.
type Message = decode.Message

func messageTransmission(message *Message) []byte {
	if message == nil {
		return nil
	}
	return message.ReaderTransmission
}

func cloneMessage(message *Message) *Message {
	if message == nil {
		return nil
	}
	clone := *message
	clone.Data = append([]byte(nil), message.Data...)
	clone.ReaderTransmission = append([]byte(nil), message.ReaderTransmission...)
	clone.Controls = append([]decode.MessageControl(nil), message.Controls...)
	return &clone
}

func equalMessages(a, b *Message) bool {
	if a == nil || b == nil {
		return a == b
	}
	return a.Variant == b.Variant &&
		bytes.Equal(a.Data, b.Data) &&
		bytes.Equal(a.ReaderTransmission, b.ReaderTransmission) &&
		equalMessageControls(a.Controls, b.Controls)
}

func equalMessageControls(a, b []decode.MessageControl) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
