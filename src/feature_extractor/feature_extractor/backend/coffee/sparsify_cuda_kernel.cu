#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

__constant__ short threshold;

template <typename scalar_t> __global__ void compute_diff_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dense_representation
) {
    const int row_idx = blockIdx.y;
    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    if(row_idx < input.size(0) && col_idx < input.size(1)-1) {
        scalar_t this_val = input[row_idx][col_idx]   > threshold ? 1 : 0;
        scalar_t next_val = input[row_idx][col_idx+1] > threshold ? 1 : 0;

        if(this_val != next_val) {
            dense_representation[row_idx][col_idx] = next_val - this_val;
        }
    }
}

template <typename scalar_t> __global__ void compute_pseudo_height_kernel(
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> crows,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> cols,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values
) {
    const int row_idx = blockIdx.y;
    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int start_idx = crows[row_idx];
    const int end_idx = crows[row_idx+1];

    const int index = start_idx + col_idx;

    if(index < end_idx-1) {
        if(values[index] < 0) { // Peak detected
            values[index] = cols[index+1] - cols[index]; // Pseudo-height
        } else { // Peak shadow end detected
            values[index] = cols[index] - cols[index+1]; // Negative pseudo-height
        }
    } else if(index == end_idx-1) { // Infinite shadow
        if(values[index] < 0) { // Peak detected
            values[index] = std::numeric_limits<scalar_t>::max();
        } else { // Peak shadow end detected
            values[index] = std::numeric_limits<scalar_t>::min();
        }
    }
}

void coffee_cuda_initialize(short cpu_threshold) {
    cudaMemcpyToSymbol(threshold, &cpu_threshold, sizeof(short), size_t(0), cudaMemcpyHostToDevice);
}

torch::Tensor coffee_cuda_sparsify(torch::Tensor input) {

    const auto num_rows = input.size(0);
    const auto num_cols = input.size(1);

    auto dense_representation = torch::zeros_like(input);

    const int threads = 1024;
    const dim3 blocks((num_cols + threads - 1) / threads, num_rows);

    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "compute_diff_kernel", ([&] {
        compute_diff_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            dense_representation.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    auto sparse_representation = dense_representation.to_sparse_csr();
    auto crows = sparse_representation.crow_indices();
    auto cols = sparse_representation.col_indices();
    auto values = sparse_representation.values();

    AT_DISPATCH_INTEGRAL_TYPES(values.type(), "compute_pseudo_height_kernel", ([&] {
        compute_pseudo_height_kernel<scalar_t><<<blocks, threads>>>(
            crows.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            cols.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));

    return sparse_representation;
}