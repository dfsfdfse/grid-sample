## burn
### grid smaple

cubecl custom wgpu runtime kernel 
Burn Implementing 4D padding with zero and bilinear interpolation for grid sampling in WGSL
The 4070s takes 5 microseconds, and its performance can be tested and viewed in the benches

### Bench
```bash
  cargo bench --bench sample
```
### Test
```bash
  cargo test test_sample -- --nocapture
```

### Performance
cube CUDA kernel
![pdf_small.svg](criterion/cube/report/pdf_small.svg)
wgsl kernel
![pdf_small.svg](criterion/wgsl/report/pdf_small.svg)