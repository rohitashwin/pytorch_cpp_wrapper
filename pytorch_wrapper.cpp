#include <torch/extension.h>
#include "matmul.hpp"

torch::Tensor custom_matmul_cpp(const torch::Tensor& a, const torch::Tensor& b) {
    // Check if the input tensors are 2D
    TORCH_CHECK(a.dim() == 2, "Input tensor 'a' must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input tensor 'b' must be 2D");
    
    // Check if the inner dimensions match
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions of 'a' and 'b' must match");

    auto a_cpu = a.contiguous().to(torch::kCPU);
    auto b_cpu = b.contiguous().to(torch::kCPU);

    float *a_ptr = a_cpu.data_ptr<float>();
    float *b_ptr = b_cpu.data_ptr<float>();
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);
    float *c_ptr = new float[m * n];
    matmul_impl(a_ptr, b_ptr, c_ptr, m, n, k);
    // Create a new tensor for the result
    torch::Tensor c = torch::empty({m, n}, torch::kFloat32);
    // Copy the result into the new tensor
    std::memcpy(c.data_ptr<float>(), c_ptr, m * n * sizeof(float));
    delete[] c_ptr;
    return c.to(a.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matmul_cpp", &custom_matmul_cpp, "Custom Matrix Multiplication");
}