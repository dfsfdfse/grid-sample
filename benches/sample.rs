use criterion::{criterion_group, criterion_main, Criterion};

fn test_sample(c: &mut Criterion) {
    use burn_grid_sample::kernel::backend::grid_sample;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::{Distribution, Tensor};
    use burn::backend::wgpu::WgpuRuntime;
    use burn_jit::JitBackend;

    type Wgpu = JitBackend<WgpuRuntime, f32, i32, u32>;
    let device = WgpuDevice::DefaultDevice;
    let v: Tensor<Wgpu, 4> = Tensor::random([1, 3, 256, 256], Distribution::Default, &device);
    let g: Tensor<Wgpu, 4> = Tensor::random([1 ,80, 80, 2], Distribution::Default, &device) * 2.0 - 1.0;

    c.bench_function("sample", |b| {
        b.iter(|| {
            grid_sample(v.clone(), g.clone());
        })
    });
}

criterion_group!(
    sample,
    test_sample,
);
criterion_main!(sample);