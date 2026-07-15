struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    blocks_x: u32,
    blocks_y: u32,
}

struct Threshold {
    anchor: vec4<f32>,
    mean: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> thresholds: array<Threshold>;
@group(0) @binding(2) var<storage, read> params: Params;

var<workgroup> partial_minimum: array<vec3<u32>, 64>;
var<workgroup> partial_maximum: array<vec3<u32>, 64>;
var<workgroup> partial_sum: array<vec3<u32>, 64>;
var<workgroup> partial_count: array<u32, 64>;

fn reduce_at(local_index: u32, offset: u32) {
    if local_index < offset {
        partial_minimum[local_index] = min(
            partial_minimum[local_index],
            partial_minimum[local_index + offset],
        );
        partial_maximum[local_index] = max(
            partial_maximum[local_index],
            partial_maximum[local_index + offset],
        );
        partial_sum[local_index] += partial_sum[local_index + offset];
        partial_count[local_index] += partial_count[local_index + offset];
    }
}

fn anchor(minimum: u32, maximum: u32) -> f32 {
    if maximum - minimum < 24u {
        return f32(minimum) / 2.0;
    }
    return f32(minimum) + 0.25 * f32(maximum - minimum);
}

@compute @workgroup_size(8, 8)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let local_index = local_id.y * 8u + local_id.x;
    let start_x = workgroup_id.x * params.block_size;
    let start_y = workgroup_id.y * params.block_size;
    let end_x = min(start_x + params.block_size, params.width);
    let end_y = min(start_y + params.block_size, params.height);
    var minimum = vec3<u32>(255u);
    var maximum = vec3<u32>(0u);
    var sum = vec3<u32>(0u);
    var count = 0u;
    for (var y = start_y + local_id.y; y < end_y; y += 8u) {
        for (var x = start_x + local_id.x; x < end_x; x += 8u) {
            let pixel = pixels[y * params.width + x];
            let value = vec3<u32>(
                pixel & 0xffu,
                (pixel >> 8u) & 0xffu,
                (pixel >> 16u) & 0xffu,
            );
            minimum = min(minimum, value);
            maximum = max(maximum, value);
            sum += value;
            count += 1u;
        }
    }
    partial_minimum[local_index] = minimum;
    partial_maximum[local_index] = maximum;
    partial_sum[local_index] = sum;
    partial_count[local_index] = count;

    workgroupBarrier();
    reduce_at(local_index, 32u);
    workgroupBarrier();
    reduce_at(local_index, 16u);
    workgroupBarrier();
    reduce_at(local_index, 8u);
    workgroupBarrier();
    reduce_at(local_index, 4u);
    workgroupBarrier();
    reduce_at(local_index, 2u);
    workgroupBarrier();
    reduce_at(local_index, 1u);
    workgroupBarrier();

    if local_index == 0u {
        let minimum_value = partial_minimum[0];
        let maximum_value = partial_maximum[0];
        let sum_value = partial_sum[0];
        let count_value = f32(partial_count[0]);
        let block_index = workgroup_id.y * params.blocks_x + workgroup_id.x;
        thresholds[block_index].anchor = vec4<f32>(
            anchor(minimum_value.x, maximum_value.x),
            anchor(minimum_value.y, maximum_value.y),
            anchor(minimum_value.z, maximum_value.z),
            0.0,
        );
        thresholds[block_index].mean = vec4<f32>(
            f32(sum_value.x) / count_value,
            f32(sum_value.y) / count_value,
            f32(sum_value.z) / count_value,
            0.0,
        );
    }
}
