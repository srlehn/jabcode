// Bitplane mask accessor: one contiguous plane per channel, 32 pixels per word.
// Same total bits as the interleaved layout, but a single-channel walk covers
// four times as many pixels per word and never loads the other two channels,
// which is the axis the interleaved layout cannot vary.

fn mask_bit(pixel: u32, channel: u32) -> u32 {
    return (masks[channel * params.plane_words + pixel / 32u] >> (pixel % 32u)) & 1u;
}
