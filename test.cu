#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Dimensions: D batches, matrices of size MxK and KxN
    const int D = 2;
    const int M = 3;
    const int K = 4;
    const int N = 5;

    // Host data initialization (batch of matrices)
    std::vector<float> h_A(D * M * K);
    std::vector<float> h_B(D * K * N);
    std::vector<float> h_C(D * M * N, 0.0f);

    // Fill A and B with some example data
    for (int i = 0; i < D * M * K; i++) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < D * K * N; i++) h_B[i] = static_cast<float>(i + 1);

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    checkCuda(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    checkCuda(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    checkCuda(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // cuBLAS uses column-major order by default, but our data is row-major.
    // We can simulate row-major multiplication by swapping matrices and transposing.
    // The batched GEMM signature:
    // C_i = alpha * op(A_i) * op(B_i) + beta * C_i

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Leading dimensions (number of rows in column-major)
    int lda = K; // A is MxK row-major, so lda=K for column-major view transposed
    int ldb = N; // B is KxN row-major, so ldb=N
    int ldc = N; // C is MxN row-major, so ldc=N

    // Arrays of device pointers to individual matrices
    std::vector<const float*> A_array(D);
    std::vector<const float*> B_array(D);
    std::vector<float*> C_array(D);

    for (int i = 0; i < D; i++) {
        A_array[i] = d_A + i * M * K;
        B_array[i] = d_B + i * K * N;
        C_array[i] = d_C + i * M * N;
    }

    float **d_A_array, **d_B_array, **d_C_array;
    checkCuda(cudaMalloc(&d_A_array, D * sizeof(float*)));
    checkCuda(cudaMalloc(&d_B_array, D * sizeof(float*)));
    checkCuda(cudaMalloc(&d_C_array, D * sizeof(float*)));

    checkCuda(cudaMemcpy(d_A_array, A_array.data(), D * sizeof(float*), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B_array, B_array.data(), D * sizeof(float*), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C_array, C_array.data(), D * sizeof(float*), cudaMemcpyHostToDevice));

    // We want to do: C_i = A_i * B_i with row-major data.
    // cuBLAS expects column-major. To handle row-major matrices:
    // We swap A and B and transpose both:
    // C_i^T = B_i^T * A_i^T

    checkCublas(cublasSgemmBatched(
        handle,
        CUBLAS_OP_N,   // B_i^T (treat B_i normally because we swap)
        CUBLAS_OP_N,   // A_i^T
        N,             // rows of B_i^T (N)
        M,             // cols of A_i^T (M)
        K,             // shared dimension
        &alpha,
        d_B_array,     // B_i
        ldb,
        d_A_array,     // A_i
        lda,
        &beta,
        d_C_array,     // C_i
        ldc,
        D));

    checkCuda(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    for (int d = 0; d < D; ++d) {
        std::cout << "Batch " << d << " result:\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << h_C[d * M * N + i * N + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    cublasDestroy(handle);

    return 0;
}

