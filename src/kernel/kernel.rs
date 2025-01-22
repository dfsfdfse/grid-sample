use cubecl::prelude::*;

#[cube(launch)]
pub fn grid_sample_kernel<F: Float>(
    input: &Tensor<F>,
    grid: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let cc = output.shape(1);
    let hh = input.shape(2);
    let ww = input.shape(3);
    let grid_h = grid.shape(1);
    let grid_w = grid.shape(2);
    let n = ABSOLUTE_POS_Z / cc;
    let c = ABSOLUTE_POS_Z % cc;
    let h = ABSOLUTE_POS_Y;
    let w = ABSOLUTE_POS_X;

    if h >= grid_h || w >= grid_w {
        return;
    }

    let grid_index = n * grid_h * grid_w + h * grid_w + w;
    let grid_index_x = grid_index * 2;
    let grid_index_y = grid_index_x + 1;
    let output_index = n * cc * grid_h * grid_w + c * grid_h * grid_w + h * grid_w + w;
    let x: F = grid[grid_index_x];
    let y: F = grid[grid_index_y];

    let x = (f32::cast_from(x) + 1.0) * f32::cast_from(ww) / 2.0 - 0.5;
    let y = (f32::cast_from(y) + 1.0) * f32::cast_from(hh) / 2.0 - 0.5;

    if x < 0.0 || y < 0.0 || x >= f32::cast_from(ww) || y >= f32::cast_from(hh) {
        output[output_index] = F::new(0.0);
        return;
    }
    let x0 = u32::cast_from(x);
    let x1 = x0 + 1;
    let y0 = u32::cast_from(y);
    let y1 = y0 + 1;

    let x0 = max::<F>(
        min::<F>(F::cast_from(x0), F::cast_from(ww - 1)),
        F::new(0.0),
    );
    let x1 = max::<F>(
        min::<F>(F::cast_from(x1), F::cast_from(ww - 1)),
        F::new(0.0),
    );
    let y0 = max::<F>(
        min::<F>(F::cast_from(y0), F::cast_from(hh - 1)),
        F::new(0.0),
    );
    let y1 = max::<F>(
        min::<F>(F::cast_from(y1), F::cast_from(hh - 1)),
        F::new(0.0),
    );

    let wa = (x1 - F::cast_from(x)) * (y1 - F::cast_from(y));
    let wb = (x1 - F::cast_from(x)) * (F::cast_from(y) - y0);
    let wc = (F::cast_from(x) - x0) * (y1 - F::cast_from(y));
    let wd = (F::cast_from(x) - x0) * (F::cast_from(y) - y0);
    let prev_index = n * cc * hh * ww + c * hh * ww;
    let input_index_a = prev_index + u32::cast_from(y0) * ww + u32::cast_from(x0);
    let input_index_b = prev_index + u32::cast_from(y1) * ww + u32::cast_from(x0);
    let input_index_c = prev_index + u32::cast_from(y0) * ww + u32::cast_from(x1);
    let input_index_d = prev_index + u32::cast_from(y1) * ww + u32::cast_from(x1);
    let a = input[input_index_a];
    let b = input[input_index_b];
    let c = input[input_index_c];
    let d = input[input_index_d];
    output[output_index] = a * wa + b * wb + c * wc + d * wd;
}

#[cube]
fn max<F: Float>(a: F, b: F) -> F {
    if a > b { a } else { b }
}

#[cube]
fn min<F: Float>(a: F, b: F) -> F {
    if a < b { a } else { b }
}
