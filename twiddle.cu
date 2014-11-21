#include<stdio.h>
#include<cuda.h>

/* Producing twiddle factors */
#define NUM_OF_X_THREADS 10
#define NUM_OF_Y_THREADS 10

__global__ void inputKernel(float *x, int N)
{
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * NUM_OF_X_THREADS + ix;

    if (idx < N)
        x[idx]  = x[idx] + (float)idx;
}

__global__ void factorKernel(float *w, int N)
{
    int ix  = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = ix * 2;
    int izx = N + idx;

    const float pi = 3.1415;
    float aw = (2.0 * pi) / (float)N;
    float arg = aw * (float)ix;

    /* Twiddle factors are symmetric along N/2. with change in sign, due to 180 degree phase change */
    if (idx < N) {
        w[idx] = cos(arg);
        w[idx + 1] = sin(arg);
        w[izx] = (-1) * w[idx];
        w[izx+1] = (-1) * w[idx + 1];
    }
}

__global__ void twiddleRealKernel(float *wr, float *w, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 0, index;

    if (idx < N) {
        if (idx == 0) {
            for (i = 0; i < N; i++)
                wr[idx * N + i] = 1;
            } else {
                wr[idx * N + 0] = 1;
                for (i = 1; i < N; i++) {
                    index = (idx * i) % N;
                    wr[idx * N + i] = w[index * 2];
                }
            }
    }
}

__global__ void twiddleImgKernel(float *wi, float *w, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i, index;

    if (idx < N) {
        if (idx == 0) {
          for (i = 0; i < N; i++)
          wi[idx * N + i] = 0;
        } else {
             wi[idx * N + 0] = 0;
             for (i = 1; i < N; i++) {
                index = (idx * i) % N;
                wi[idx * N + i] = w[index * 2 + 1];

             }
        }
    }
}

int main(int agrc, char** argv)
{
    float *x, *w, *w_r, *w_i;
    float *d_x, *d_w, *dw_r, *dw_i;

    int N = 10000, n = N/2;

    x = (float *)malloc(N * sizeof(float));
    w = (float *)malloc(2 * N * sizeof(float));
    w_r = (float *)malloc(N * N * sizeof(float));
    w_i = (float *)malloc(N * N * sizeof(float));
    dim3 numberOfThreads(NUM_OF_X_THREADS, NUM_OF_Y_THREADS);
    dim3 numberOfBlocks( (100 + NUM_OF_X_THREADS -1)/NUM_OF_X_THREADS,
                         (100 + NUM_OF_Y_THREADS - 1)/NUM_OF_Y_THREADS );

    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_w, 2 * N * sizeof(float));
    cudaMalloc((void **)&dw_r, N * N * sizeof(float));
    cudaMalloc((void **)&dw_i, N * N * sizeof(float));

    cudaMemset(d_x, 0, N * sizeof(float));
    cudaMemset(d_w, 0, 2 * N * sizeof(float));
    cudaMemset(dw_r, 0, N * N * sizeof(float));
    cudaMemset(dw_i, 0, N * N * sizeof(float));

    inputKernel<<<numberOfBlocks, numberOfThreads>>>(d_x, N);
    cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n",x[100]);
    // Calculating factor
    factorKernel<<<n/512, 512>>>(d_w, (float)N);
    cudaMemcpy(w, d_w, 2 * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f %f\n", w[5], w[10005]);

    // Calculating twiddle real matrix
    twiddleRealKernel<<<n/512, 512>>>(dw_r, d_w, N);
    cudaMemcpy(w_r, dw_r, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculating twiddle imaginary matrix
    twiddleImgKernel<<<n/512, 512>>>(dw_i, d_w, N);
    cudaMemcpy(w_i, dw_i, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    /*  int i,j;
        for(i = 0; i < 50; i++)
        {
            for(j = 0; j < 50; j++) {
                printf("%f \t", w_r[i*N + j]);
            }
            printf("\n");
        }
      printf("*********************************************************************************\n");
      for(i = 0; i < 50; i++) {
        for(j = 0; j < 50; j++) {
          printf("%f \t", w_i[i*N + j]);
        }
        printf("\n");
      }
*/
  return 0;
}




