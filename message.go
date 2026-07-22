package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/read"
)

// MessageControlKind identifies a structured message control that is not a
// literal byte in Message.Data.
type MessageControlKind uint8

const (
	MessageControlECI MessageControlKind = iota + 1
	MessageControlFNC1Start
	MessageControlFNC1Separator
	MessageControlFNC1End
	MessageControlISO15434Start
	MessageControlISO15434End
)

// MessageControl records a control at an offset in Message.Data. Assignment is
// set only for MessageControlECI. A FNC1 separator's offset points at the GS
// byte inserted into Data.
type MessageControl struct {
	Kind       MessageControlKind
	Offset     int
	Assignment int
}

// Message contains raw decoded data and the standards-facing reader
// transmission produced from the same corrected message bits. Data excludes
// symbology identifiers and ECI escape fields and restores literal
// backslashes. Controls preserves the non-data structure.
type Message struct {
	Data               []byte
	ReaderTransmission []byte
	Controls           []MessageControl
}

// DecodeMessage decodes img once and returns raw application data alongside
// the reader transmission. Decode remains the standards-facing shorthand for
// Message.ReaderTransmission.
func DecodeMessage(img image.Image) (Message, error) {
	return guardImage(func() (Message, error) {
		message, err := read.DecodeMessage(img)
		if err != nil {
			return Message{}, err
		}
		return publicMessage(message), nil
	})
}

func publicMessage(message *read.Message) Message {
	if message == nil {
		return Message{}
	}
	result := Message{
		Data:               append([]byte(nil), message.Data...),
		ReaderTransmission: append([]byte(nil), message.ReaderTransmission...),
		Controls:           make([]MessageControl, len(message.Controls)),
	}
	for i, control := range message.Controls {
		result.Controls[i] = MessageControl{
			Kind: publicMessageControlKind(control.Kind), Offset: control.Offset,
			Assignment: control.Assignment,
		}
	}
	return result
}

func publicMessageControlKind(kind decode.MessageControlKind) MessageControlKind {
	switch kind {
	case decode.MessageControlECI:
		return MessageControlECI
	case decode.MessageControlFNC1Start:
		return MessageControlFNC1Start
	case decode.MessageControlFNC1Separator:
		return MessageControlFNC1Separator
	case decode.MessageControlFNC1End:
		return MessageControlFNC1End
	case decode.MessageControlISO15434Start:
		return MessageControlISO15434Start
	case decode.MessageControlISO15434End:
		return MessageControlISO15434End
	default:
		return 0
	}
}
