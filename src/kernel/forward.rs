use std::marker::PhantomData;

use crate::kernel::backend::{Backend, Backend1};
use burn::tensor::ops::FloatTensor;
use burn::{
    backend::wgpu::{
        BoolElement, FloatElement, IntElement, JitBackend, JitTensor, KernelSource, SourceKernel,
        SourceTemplate, WgpuRuntime, build_info, into_contiguous, kernel_source,
    },
    tensor::Shape,
};
use burn_jit::JitRuntime;
use cubecl::{CubeCount, CubeDim};
use derive_new::new;
use crate::kernel::kernel::grid_sample_kernel;

kernel_source!(GridSampleRaw, "./kernel.wgsl");

#[derive(new, Debug)]
struct GridSample<E: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<E>,
}

impl<E: FloatElement> KernelSource for GridSample<E> {
    fn source(&self) -> SourceTemplate {
        GridSampleRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info(self.cube_dim)
    }
}

impl<F: FloatElement, I: IntElement, B: BoolElement> Backend for JitBackend<WgpuRuntime, F, I, B> {
    fn grid_sample(input: FloatTensor<Self>, grid: FloatTensor<Self>) -> FloatTensor<Self> {
        assert_eq!(grid.shape.dims[3], 2, "Grid sample only supports 2D grids");
        assert_eq!(grid.shape.dims[0], input.shape.dims[0], "Grid sample only supports same batch size");

        let input = into_contiguous(input);
        let grid = into_contiguous(grid);
        let shape_out: Vec<usize> = vec![
            input.shape.dims[0],
            input.shape.dims[1],
            grid.shape.dims[1],
            grid.shape.dims[2],
        ];

        let shape = Shape::from(shape_out);
        let buffer = input.client.empty(shape.num_elements() * core::mem::size_of::<F>());
        let output = JitTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape,
            buffer,
            F::dtype(),
        );
        let nr = grid.shape.dims[1];
        let nc = grid.shape.dims[2];
        let cx = (nr as f32 / 16.0).ceil() as u32;
        let cy = (nc as f32 / 16.0).ceil() as u32;
        let cube_dim = CubeDim {
            x: 16,
            y: 16,
            z: 1,
        };
        let cube_count = CubeCount::Static(cx, cy, (input.shape.dims[0] * input.shape.dims[1]) as u32);
        let kernel = GridSample::<F>::new(cube_dim);
        let info = build_info::<_, F>(&[&input, &grid, &output]);
        let info_handle = input.client.create(bytemuck::cast_slice(&info));
        input.client.execute(
            Box::new(SourceKernel::new(kernel, cube_dim)),
            cube_count,
            vec![
                input.handle.binding(),
                grid.handle.binding(),
                output.handle.clone().binding(),
                info_handle.binding(),
            ],
        );
        
        output
    }

}


impl<R: JitRuntime ,F: FloatElement, I: IntElement, B: BoolElement> Backend1 for JitBackend<R, F, I, B> {
    fn grid_sample(input: FloatTensor<Self>, grid: FloatTensor<Self>) -> FloatTensor<Self> {
        assert_eq!(grid.shape.dims[3], 2, "Grid sample only supports 2D grids");
        assert_eq!(grid.shape.dims[0], input.shape.dims[0], "Grid sample only supports same batch size");

        let input = into_contiguous(input);
        let grid = into_contiguous(grid);
        let shape_out: Vec<usize> = vec![
            input.shape.dims[0],
            input.shape.dims[1],
            grid.shape.dims[1],
            grid.shape.dims[2],
        ];

        let shape = Shape::from(shape_out);
        let buffer = input.client.empty(shape.num_elements() * core::mem::size_of::<F>());
        let output = JitTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape,
            buffer,
            F::dtype(),
        );
        let nr = grid.shape.dims[1];
        let nc = grid.shape.dims[2];
        let cx = (nr as f32 / 16.0).ceil() as u32;
        let cy = (nc as f32 / 16.0).ceil() as u32;
        let cube_dim = CubeDim {
            x: 16,
            y: 16,
            z: 1,
        };
        let cube_count = CubeCount::Static(cx, cy, (input.shape.dims[0] * input.shape.dims[1]) as u32);

        grid_sample_kernel::launch::<F, I, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<F>(1),
            grid.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        output
    }
}
