struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    blocks_x: u32,
    blocks_y: u32,
    flags: u32,
    fixed_r: f32,
    fixed_g: f32,
    fixed_b: f32,
}

@group(0) @binding(0) var<storage, read> input_masks: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_masks: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

var<workgroup> raw_tile: array<u32, 144>;
var<workgroup> horizontal_tile: array<u32, 96>;

fn majority(a: u32, b: u32, c: u32, d: u32, e: u32, bit: u32) -> u32 {
    let votes = ((a >> bit) & 1u) +
        ((b >> bit) & 1u) +
        ((c >> bit) & 1u) +
        ((d >> bit) & 1u) +
        ((e >> bit) & 1u);
    return select(0u, 1u << bit, votes > 2u);
}

fn majority_mask(a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
    return majority(a, b, c, d, e, 0u) |
        majority(a, b, c, d, e, 1u) |
        majority(a, b, c, d, e, 2u);
}

@compute @workgroup_size(8, 8)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let local_index = local_id.y * 8u + local_id.x;
    let base_x = i32(group_id.x * 8u);
    let base_y = i32(group_id.y * 8u);
    let width = i32(params.width);
    let height = i32(params.height);

    var load_index = local_index;
    loop {
        if load_index >= 144u {
            break;
        }
        let tile_x = i32(load_index % 12u);
        let tile_y = i32(load_index / 12u);
        let source_x = clamp(base_x + tile_x - 2, 0, width - 1);
        let source_y = clamp(base_y + tile_y - 2, 0, height - 1);
        raw_tile[load_index] = input_masks[u32(source_y) * params.width + u32(source_x)];
        load_index = load_index + 64u;
    }
    workgroupBarrier();

    var horizontal_index = local_index;
    loop {
        if horizontal_index >= 96u {
            break;
        }
        let tile_x = horizontal_index % 8u;
        let tile_y = horizontal_index / 8u;
        let source_x = base_x + i32(tile_x);
        let source_y = base_y + i32(tile_y) - 2;
        let center = tile_y * 12u + tile_x + 2u;
        var filtered = raw_tile[center];
        if source_x >= 2 && source_x + 2 < width && source_y >= 2 && source_y + 2 < height {
            filtered = majority_mask(
                raw_tile[center - 2u],
                raw_tile[center - 1u],
                raw_tile[center],
                raw_tile[center + 1u],
                raw_tile[center + 2u],
            );
        }
        horizontal_tile[horizontal_index] = filtered;
        horizontal_index = horizontal_index + 64u;
    }
    workgroupBarrier();

    let output_x = base_x + i32(local_id.x);
    let output_y = base_y + i32(local_id.y);
    if output_x >= width || output_y >= height {
        return;
    }
    let output_index = u32(output_y) * params.width + u32(output_x);
    let raw_center = (local_id.y + 2u) * 12u + local_id.x + 2u;
    if output_x < 2 || output_x + 2 >= width || output_y < 2 || output_y + 2 >= height {
        output_masks[output_index] = raw_tile[raw_center];
        return;
    }
    let center = local_id.y * 8u + local_id.x;
    output_masks[output_index] = majority_mask(
        horizontal_tile[center],
        horizontal_tile[center + 8u],
        horizontal_tile[center + 16u],
        horizontal_tile[center + 24u],
        horizontal_tile[center + 32u],
    );
}
