package jabcode

func isoReaderTransmission(data []byte) []byte {
	out := make([]byte, 0, len(data)+3)
	out = append(out, ']', 'j', '1')
	return append(out, data...)
}

func isoReaderTransmissionString(data string) string {
	return string(isoReaderTransmission([]byte(data)))
}
