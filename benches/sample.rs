use burn::backend::cuda_jit::CudaDevice;
use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::cuda::CudaRuntime;

fn test_wgsl(c: &mut Criterion) {
    use grid_sample::kernel::backend::grid_sample;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::{Distribution, Tensor};
    use burn::backend::wgpu::WgpuRuntime;
    use burn_jit::JitBackend;

    type Wgpu = JitBackend<WgpuRuntime, f32, i32, u32>;
    let device = WgpuDevice::DefaultDevice;
    let v: Tensor<Wgpu, 4> = Tensor::random([1, 3, 256, 256], Distribution::Default, &device);
    let g: Tensor<Wgpu, 4> = Tensor::random([1 ,80, 80, 2], Distribution::Default, &device) * 2.0 - 1.0;

    c.bench_function("wgsl", |b| {
        b.iter(|| {
            grid_sample(v.clone(), g.clone());
        })
    });
}

fn test_cube(c: &mut Criterion) {
    use burn::tensor::{Distribution, Tensor};
    use burn_jit::JitBackend;
    use grid_sample::kernel::backend::grid_sample_cube;
    
    type Wgpu = JitBackend<CudaRuntime, f32, i32, u32>;
    let device = CudaDevice::new(0);
    let v: Tensor<Wgpu, 4> = Tensor::random([1, 3, 256, 256], Distribution::Default, &device);
    let g: Tensor<Wgpu, 4> = Tensor::random([1 ,80, 80, 2], Distribution::Default, &device) * 2.0 - 1.0;

    c.bench_function("cube", |b| {
        b.iter(|| {
            grid_sample_cube(v.clone(), g.clone());
        })
    });
}

criterion_group!(
    sample,
    test_wgsl,
    test_cube,
);
criterion_main!(sample);