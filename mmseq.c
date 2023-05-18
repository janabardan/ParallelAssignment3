#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiplication(float** A, float** B, float** result, int m, int n, int k) {
    int i, j, p;

    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            result[i][j] = 0.0;
            for (p = 0; p < n; p++) {
                result[i][j] += A[i][p] * B[p][j];
            }
        }
    }
}

int main() {
    int m = 1000;   // Number of rows in matrix A
    int n = 2000;   // Number of columns in matrix A
    int k = 3000;   // Number of columns in matrix B

    // Seed the random number generator
    srand(time(NULL));

    // Matrix A
    float** A = (float**)malloc(m * sizeof(float*));
    int i, j;
    for (i = 0; i < m; i++) {
        A[i] = (float*)malloc(n * sizeof(float));
        for (j = 0; j < n; j++) {
            A[i][j] = (float)rand() / RAND_MAX;  
        }
    }

    // Matrix B
    float** B = (float**)malloc(n * sizeof(float*));
    for (i = 0; i < n; i++) {
        B[i] = (float*)malloc(k * sizeof(float));
        for (j = 0; j < k; j++) {
            B[i][j] = (float)rand() / RAND_MAX;  
        }
    }

    // Result matrix
    float** result = (float**)malloc(m * sizeof(float*));
    for (i = 0; i < m; i++) {
        result[i] = (float*)malloc(k * sizeof(float));
    }
   clock_t start_time = clock();
    matrix_multiplication(A, B, result, m, n, k);
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the execution time
    printf("Elapsed Time: %.2f seconds\n", elapsed_time);

    return 0;
}
