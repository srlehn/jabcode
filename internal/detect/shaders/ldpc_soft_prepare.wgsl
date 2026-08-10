// Converts the resident hard-LDPC verdicts into two indirect dispatches. A clean
// payload writes zero workgroups, so neither reliability ranking nor min-sum
// spends device work on the ordinary hard-success path.

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_BLOCKS: u32 = 4u;

@group(0) @binding(0) var<storage, read> payload_params: array<u32>;
@group(0) @binding(1) var<storage, read> ldpc_params: array<u32>;
@group(0) @binding(2) var<storage, read> net: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let blocks = ldpc_params[PARAM_BLOCKS];
    var retry = false;
    for (var block = 0u; block < blocks; block += 1u) {
        retry = retry || atomicLoad(&net[block]) == 1u;
    }
    indirect[0] = select(0u, payload_params[PARAM_SIDE_X] * payload_params[PARAM_SIDE_Y], retry);
    indirect[1] = 1u;
    indirect[2] = 1u;
    indirect[3] = select(0u, 2u, retry);
    indirect[4] = 1u;
    indirect[5] = 1u;
    indirect[6] = select(0u, blocks, retry);
    indirect[7] = 1u;
    indirect[8] = 1u;
}
