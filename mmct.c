#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

__global__ void matrixMultiplication(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    for (int t = 0; t < n; t += TILE_SIZE) {
        __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

        if (row < m && t + threadIdx.x < n) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (t + threadIdx.y < n && col < k) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * k + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

int main() {
   int m = 1000;   // Number of rows in matrix A
    int n = 2000;   // Number of columns in matrix A
    int k = 3000;   // Number of columns in matrix B

    // Seed the random number generator
    srand(time(NULL));

    // Matrix A
    float* A = (float*)malloc(m * n * sizeof(float));
    for (int i = 0; i < m * n; i++) {
        A[i] = (float)rand() / RAND_MAX;  // Generate a random float value between 0 and 1
    }

    // Matrix B
    float* B = (float*)malloc(n * k * sizeof(float));
    for (int i = 0; i < n * k; i++) {
        B[i] = (float)rand() / RAND_MAX;  // Generate a random float value between 0 and 1
    }

    // Result matrix
    float* C = (float*)malloc(m * k * sizeof(float));

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    int size_A = m * n * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * k * sizeof(float);
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Transfer data from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 gridSize((k - 1) / TILE_SIZE + 1, (m - 1) / TILE_SIZE + 1, 1);
    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);

    // Launch the kernel
    clock_t start_time = clock();
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    clock_t end_time = clock();

    // Copy the result from device to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print the execution time
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution Time: %.2f seconds\n", execution_time);

    // Free host and device memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
