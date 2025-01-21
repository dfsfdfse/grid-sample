use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorPrimitive};

pub trait Backend: burn::tensor::backend::Backend {
    fn grid_sample(input: FloatTensor<Self>, grid: FloatTensor<Self>) -> FloatTensor<Self>;
}

pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

pub fn grid_sample<B: Backend>(input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
    Tensor::from_primitive(TensorPrimitive::Float(B::grid_sample(
        input.into_primitive().tensor(),
        grid.into_primitive().tensor(),
    )))
}