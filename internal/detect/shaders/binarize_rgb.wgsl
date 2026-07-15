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

struct Threshold {
    anchor: vec4<f32>,
    mean: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read> thresholds: array<Threshold>;
@group(0) @binding(2) var<storage, read_write> masks: array<u32>;
@group(0) @binding(3) var<storage, read> params: Params;

fn threshold_at(x: u32, y: u32, anchor: bool) -> vec3<f32> {
    if (params.flags & 1u) != 0u {
        return vec3<f32>(params.fixed_r, params.fixed_g, params.fixed_b);
    }

    let bs = f32(params.block_size);
    let fx = (f32(x) + 0.5) / bs - 0.5;
    let fy = (f32(y) + 0.5) / bs - 0.5;
    let floor_x = floor(fx);
    let floor_y = floor(fy);
    let bx = i32(floor_x);
    let by = i32(floor_y);
    let tx = fx - floor_x;
    let ty = fy - floor_y;
    let x0 = u32(clamp(bx, 0, i32(params.blocks_x) - 1));
    let x1 = u32(clamp(bx + 1, 0, i32(params.blocks_x) - 1));
    let y0 = u32(clamp(by, 0, i32(params.blocks_y) - 1));
    let y1 = u32(clamp(by + 1, 0, i32(params.blocks_y) - 1));

    var top_left: vec3<f32>;
    var top_right: vec3<f32>;
    var bottom_left: vec3<f32>;
    var bottom_right: vec3<f32>;
    if anchor {
        top_left = thresholds[y0 * params.blocks_x + x0].anchor.xyz;
        top_right = thresholds[y0 * params.blocks_x + x1].anchor.xyz;
        bottom_left = thresholds[y1 * params.blocks_x + x0].anchor.xyz;
        bottom_right = thresholds[y1 * params.blocks_x + x1].anchor.xyz;
    } else {
        top_left = thresholds[y0 * params.blocks_x + x0].mean.xyz;
        top_right = thresholds[y0 * params.blocks_x + x1].mean.xyz;
        bottom_left = thresholds[y1 * params.blocks_x + x0].mean.xyz;
        bottom_right = thresholds[y1 * params.blocks_x + x1].mean.xyz;
    }
    let top = top_left + (top_right - top_left) * tx;
    let bottom = bottom_left + (bottom_right - bottom_left) * tx;
    return top + (bottom - top) * ty;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let index = y * params.width + x;
    let packed = pixels[index];
    let rgb = vec3<u32>(packed & 0xffu, (packed >> 8u) & 0xffu, (packed >> 16u) & 0xffu);
    let white_threshold = threshold_at(x, y, false);
    var black_threshold = white_threshold;
    if (params.flags & 2u) != 0u {
        black_threshold = threshold_at(x, y, true);
    }

    if f32(rgb.x) < black_threshold.x &&
       f32(rgb.y) < black_threshold.y &&
       f32(rgb.z) < black_threshold.z {
        masks[index] = 0u;
        return;
    }

    var minimum = rgb.x;
    var middle = rgb.y;
    var maximum = rgb.z;
    var i_min = 0u;
    var i_mid = 1u;
    var i_max = 2u;
    if minimum > maximum {
        let swap_value = minimum;
        minimum = maximum;
        maximum = swap_value;
        let swap_index = i_min;
        i_min = i_max;
        i_max = swap_index;
    }
    if minimum > middle {
        let swap_value = minimum;
        minimum = middle;
        middle = swap_value;
        let swap_index = i_min;
        i_min = i_mid;
        i_mid = swap_index;
    }
    if middle > maximum {
        let swap_value = middle;
        middle = maximum;
        maximum = swap_value;
        let swap_index = i_mid;
        i_mid = i_max;
        i_max = swap_index;
    }

    let rg = i32(rgb.x) - i32(rgb.y);
    let rb = i32(rgb.x) - i32(rgb.z);
    let gb = i32(rgb.y) - i32(rgb.z);
    let pair_variance = u32(rg * rg + rb * rb + gb * gb);
    let near_gray = 625u * pair_variance < 36u * maximum * maximum;
    if near_gray &&
       f32(rgb.x) > white_threshold.x &&
       f32(rgb.y) > white_threshold.y &&
       f32(rgb.z) > white_threshold.z {
        masks[index] = 7u;
        return;
    }

    var mask = 1u << i_max;
    if middle > 0u && (minimum == 0u || middle * middle > maximum * minimum) {
        mask = mask | (1u << i_mid);
    }
    masks[index] = mask;
}
