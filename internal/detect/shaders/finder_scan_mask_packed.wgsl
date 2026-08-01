// Interleaved mask accessor: three channel bits per pixel, eight pixels per
// word, which is what the resident binarizer already writes. One word covers
// eight consecutive pixels of all three channels, so a single-channel walk
// touches a word every eight samples and reads three times the bits it wants.

fn mask_bit(pixel: u32, channel: u32) -> u32 {
    return (masks[pixel / 8u] >> ((pixel % 8u) * 3u + channel)) & 1u;
}
