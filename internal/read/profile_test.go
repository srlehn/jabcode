package read

func isoPayload(data []byte) []byte {
	out := make([]byte, 0, len(data)+3)
	out = append(out, ']', 'j', '1')
	return append(out, data...)
}
