package read

// primaryCorrection is the format-specific payload half of an already
// sampled and metadata-interpreted primary symbol.
type primaryCorrection interface {
	CorrectPayload() int
}
