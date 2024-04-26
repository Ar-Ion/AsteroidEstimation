#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void coffee_cuda_initialize(short threshold);
torch::Tensor coffee_cuda_sparsify(torch::Tensor input);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor coffee_sparsify(torch::Tensor input) {
  CHECK_INPUT(input);
  return coffee_cuda_sparsify(input);
}

void coffee_initialize(short threshold) {
  coffee_cuda_initialize(threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &coffee_initialize, "COFFEE initialize (CUDA)");
  m.def("sparsify", &coffee_sparsify, "COFFEE sparsify (CUDA)");
}