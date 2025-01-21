@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> grid: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read_write> info: array<u32>;

const BLOCK_SIZE = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let dim = info[0];
    // N : batches C: channels H: height W: width grid_H: grid height grid_W: grid width
    let N = info[5u * dim + 1u];
    let C = info[3u * dim + 2u];
    let H = info[4u * dim - 1u];
    let W = info[4u * dim ];
    let grid_H = info[6u * dim - 1u];
    let grid_W = info[6u * dim];
    let batch = global_id.z;
    let n = batch % N;
    let c = batch / N;
    let h = workgroup_id.y * BLOCK_SIZE + local_idx % BLOCK_SIZE;
    let w = workgroup_id.x * BLOCK_SIZE + local_idx / BLOCK_SIZE;
   
    if h >= grid_H || w >= grid_W {
        return;
    }
    
    let grid_index = n * grid_H * grid_W + h * grid_W + w;
    
    let grid_index_x = grid_index * 2u;
    let grid_index_y = grid_index_x + 1u;

    var x = grid[grid_index_x];
    var y = grid[grid_index_y];
    let a = input[1];

    x = (x + 1.0) * f32(W) / 2.0 - 0.5;
    y = (y + 1.0) * f32(H) / 2.0 - 0.5;
    let output_index = n * C * grid_H * grid_W + c * grid_H * grid_W + h * grid_W + w;
    if x < 0.0 || x >= f32(W) || y < 0.0 || y >= f32(H) {
        output[output_index] = 0.0;
        return;
    }
    
    /// bilinear interpolation
    var x0 = u32(x);
    var x1 = x0 + 1u;
    var y0 = u32(y);
    var y1 = y0 + 1u;
    
    x0 = min(x0, W - 1u);
    x1 = min(x1, W - 1u);
    y0 = min(y0, H - 1u);
    y1 = min(y1, H - 1u);
    
    let wa = (f32(x1) - x) * (f32(y1) - y);
    let wb = (f32(x1) -x) * (y - f32(y0));
    let wc = (x - f32(x0)) * (f32(y1) - y);
    let wd = (x - f32(x0)) * (y - f32(y0));
    
    let prev_index = n * C * H * W + c * H * W;
    let input_index_a = prev_index + y0 * W + x0;
    let input_index_b = prev_index + y1 * W + x0;
    let input_index_c = prev_index + y0 * W + x1;
    let input_index_d = prev_index + y1 * W + x1;

    let value_a = input[input_index_a];
    let value_b = input[input_index_b];
    let value_c = input[input_index_c];
    let value_d = input[input_index_d];

    let result = wa * value_a + wb * value_b + wc * value_c + wd * value_d;
    output[output_index] = result;
}
