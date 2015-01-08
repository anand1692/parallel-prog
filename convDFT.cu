#include<stdio.h>
#include<cuda.h>

#define NUM_OF_X_THREADS 16
#define NUM_OF_Y_THREADS 16
#define TOTAL_THREADS (NUM_OF_X_THREADS * NUM_OF_Y_THREADS)
#define TILE_WIDTH 16

/* Kernel to take input signal 1 
 * f(x) = x; where x = 0 to n-1
 */
__global__ void inputKernel(float *x, int n, int N)
{
	int ix   = blockIdx.x * blockDim.x + threadIdx.x,i;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * NUM_OF_X_THREADS + ix;

	if (idx < N)
	{
		if (idx < n)
		{
    		x[idx*N]  = (float)idx;
		}
		else 
		{
			x[idx] = 0;
		}
	
		for(i=1;i<N;i++)
		{
			x[idx*N + i] = 0;
		}
	}
	
}

/* Kernel to take input signal 2 
 * f(x) = x*2 - x^2; where x = 0 to n-1
 */
__global__ void inputKernel2(float *x, int n, int N)
{
	int ix   = blockIdx.x * blockDim.x + threadIdx.x,i;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * NUM_OF_X_THREADS + ix;

	if (idx < N)
	{
		if (idx < n)
		{
    		x[idx*N]  = ((float)idx * 2) - ((float)idx * (float)idx);
		}
		else
		{
			x[idx] = 0;
		}
		for(i=1;i<N;i++)
		{
			x[idx*N + i] = 0;
		}
	}
}

/* Kernel to generate the twiddle factors 
 * Let twiddle factors be denoted by w. 
 * w = e^(2*pi/N) * k * n; where n = 0 to N-1 and k = 0 to N-1
 * In w, the real and imaginary part are stored together. 
 * Hence, one number actually takes two index positions. Thus, w has size 2N
 */
__global__ void factorKernel(float *w, int N)
{
	int ix  = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = ix * 2;
	int izx = N + idx;
	
	const float pi = 3.141592653589793238462643383;
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

/* Kernel to arrange real part of twiddle factors in 2D : Cos(theta)
 * Let the real part of w be denoted by wr. For k*n = 1 -> wr = wr1, k*n = 2 -> wr = wr2.
 * The real twiddle matrix to take the DFT looks as below:
 *
 * 		1   1       1        1    ...  1
 *		1  wr1     wr2      wr3   ... wr(N-1) 
 *		1  wr2     wr4      wr6   ... wr(N-2)
 *		1  wr3     wr6      wr9   ... wr(N-3)
 *		.
 * 		.
 *		.
 *		1 wr(N-1) wr(N-2) wr(N-3) ... wr1
*/
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

/* Kernel to arrange imaginary part of twiddle factors in 2D : -iSin(theta)
 * Let the real part of w be denoted by wi. For k*n = 1 -> wi = wi1, k*n = 2 -> wi = wi2.
 * The imaginary twiddle matrix to take the DFT looks as below:
 *
 * 		0   0       0        0    ...  0
 *		0  wi1     wi2      wi3   ... wi(N-1) 
 *		0  wi2     wi4      wi6   ... wi(N-2)
 *		0  wi3     wi6      wi9   ... wi(N-3)
 *		.
 * 		.
 *		.
 *		0 wi(N-1) wi(N-2) wi(N-3) ... wi1
*/
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
				wi[idx * N + i] = (-1) * w[index * 2 + 1];		
			}
		}
	}	
} 

/* Kernel to arrange imaginary part of twiddle factors in 2D for taking IDFT : +iSin(theta)
 * The imaginary twiddle matrix to take IDFT is negative of imaginary twiddle matrix to take DFT
 * Let imaginary twiddle matrix to take IDFT be wi2, then
 * wi2 = (-1) * wi	
 */
__global__ void twiddleImgKernelIDFT(float *wi, float *w, int N)
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

/* Kernel to take the dot product of two matrices and storing the result 
 * in resultant matrix.
 * ab[i] = a[i] . b[i]
 */
__global__ void dotProdKernel(float *a, float *b, float *ab, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if( (idx*N) < (N*N) ) {
		ab[idx * N] = a[idx *N] * b[idx * N];
	}
}

/* Kernel to multiply to input matrices and storing the result in resultant matrix
 * The data from the two matrices is accessed in tiles of width TILE_WIDTH.	
*/
__global__ void multKernel(float *a, float *b, float *ab, int width)
{
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	// allocate tiles in __shared__ memory
	__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

	// calculate the row & col index to identify element to work on
	int row = by*blockDim.y + ty;
	int col = bx*blockDim.x + tx;
	float result = 0; 

	// loop over the tiles of the input in phases
	for(int p = 0; p < width/TILE_WIDTH; ++p)
	{
		// collaboratively load tiles into shared memory: row-wise and column wise respectively
		s_a[ty][tx] = a[row*width + (p*TILE_WIDTH + tx)];
		s_b[ty][tx] = b[(p*TILE_WIDTH + ty)*width + col];
		__syncthreads();

		// dot product between row of s_a and col of s_b
		for(int k = 0; k < TILE_WIDTH; ++k)
			result += s_a[ty][k] * s_b[k][tx];
		__syncthreads();	
	}
 	ab[row*width+col] = result;
}

/* Simple kernel to add elements of two matrices.
 * In this case, we need to just add the first column of the two matrices
 * as all other elements will be 0.
*/
__global__ void addMat(float *a, float *b, float *add, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if((idx*N) < (N*N))
    	add[idx * N] = a[idx *N] + b[idx * N];
}

/* Simple kernel to subtract elements of two matrices.
 * In this case, we need to just subtract the first column of the two matrices
 * as all other elements will be 0.
*/
__global__ void subMat(float *a, float *b, float *sub, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if((idx*N) < (N*N))
    	sub[idx * N] = a[idx * N] - b[idx * N];
}

/* Simple kernel to divide elements of matrix by N.
 * In this case, we need to just divide the first column of the two matrices
 * as all other elements will be 0.
*/
__global__ void divMat(float *a, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if((idx*N) < (N*N))
    	a[idx *N] /= N;
}

/* Main function */
int main(int agrc, char** argv)
{
	int n, i;
	n = strtol(argv[1], 0, 10);	

	/* Final Padding for convolution and multiplication */
	int N = 2*n + (TILE_WIDTH - (2*n)%TILE_WIDTH);

	/* Variables on Host */
	float realRes[N][N], imgRes[N][N];	/* Resultant convolution matrix - real & img */
	float x[N][N], sig2[N][N];			/* Two Input signals */

	/* Variables on Device */
	float *ddft_realRes, *ddft_imgRes, *d_realRes1, *d_realRes2, *d_realRes, *d_imgRes1, *d_imgRes2, *d_imgRes;
	float *d_w, *dw_r, *dw_i, *d_realProd, *d_imgProd, *d_realImg2, *d_imgReal2;
	float *d_x, *ddft_r, *ddft_i;
	float *d_sig2, *ddft2_r, *ddft2_i; 

	/* Streams */
	cudaStream_t d_x_Stream, d_sig2_Stream,d_w_Stream, dw_r_Stream, dw_i_Stream, ddft_r_Stream, ddft_i_Stream;
	cudaStream_t ddft2_i_Stream, ddft2_r_Stream, d_realProd_Stream, d_imgProd_Stream;
    cudaStream_t d_realImg2_Stream, d_imgReal2_Stream, ddft_realRes_Stream, ddft_imgRes_Stream, d_realRes1_Stream;
	cudaStream_t d_realRes2_Stream, d_realRes_Stream, d_imgRes1_Stream, d_imgRes2_Stream, d_imgRes_Stream;
	cudaStream_t dw_i2_Stream;
	
	/* Creating streams */
	cudaStreamCreate(&d_x_Stream);
	cudaStreamCreate(&d_sig2_Stream);
	cudaStreamCreate(&d_w_Stream);
	cudaStreamCreate(&dw_r_Stream);
	cudaStreamCreate(&dw_i_Stream);
	cudaStreamCreate(&ddft_r_Stream);
	cudaStreamCreate(&ddft_i_Stream);
	cudaStreamCreate(&ddft2_i_Stream);
	cudaStreamCreate(&ddft2_r_Stream);
	cudaStreamCreate(&d_realProd_Stream);
	cudaStreamCreate(&d_imgProd_Stream);
	cudaStreamCreate(&d_realImg2_Stream);
	cudaStreamCreate(&d_imgReal2_Stream);
	cudaStreamCreate(&ddft_realRes_Stream);
	cudaStreamCreate(&ddft_imgRes_Stream);
	cudaStreamCreate(&d_realRes1_Stream);
	cudaStreamCreate(&d_realRes2_Stream);
	cudaStreamCreate(&d_realRes_Stream);
	cudaStreamCreate(&d_imgRes1_Stream);
	cudaStreamCreate(&d_imgRes2_Stream);
	cudaStreamCreate(&d_imgRes_Stream);
	cudaStreamCreate(&dw_i2_Stream);
	
	/* Timer */
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 numberOfThreads(NUM_OF_X_THREADS, NUM_OF_Y_THREADS);
	dim3 numberOfBlocks( (TOTAL_THREADS + NUM_OF_X_THREADS -1)/NUM_OF_X_THREADS, 
						 (TOTAL_THREADS + NUM_OF_Y_THREADS - 1)/NUM_OF_Y_THREADS );

	/* Timer starts */
	cudaEventRecord(start, 0);

	/* Allocating memory on device */
	cudaMalloc((void **)&ddft_realRes, N * N * sizeof(float));
	cudaMalloc((void **)&ddft_imgRes, N * N * sizeof(float));
	cudaMalloc((void **)&d_realRes1, N * N * sizeof(float));
	cudaMalloc((void **)&d_realRes2, N * N * sizeof(float));
	cudaMalloc((void **)&d_realRes, N * N * sizeof(float));
	cudaMalloc((void **)&d_imgRes1, N * N * sizeof(float));
	cudaMalloc((void **)&d_imgRes2, N * N * sizeof(float));
	cudaMalloc((void **)&d_imgRes, N * N * sizeof(float));

	cudaMalloc((void **)&d_w, 2 * N * sizeof(float));
	cudaMalloc((void **)&dw_r, N * N * sizeof(float));
	cudaMalloc((void **)&dw_i, N * N * sizeof(float));
	cudaMalloc((void **)&d_realProd, N * N * sizeof(float));
	cudaMalloc((void **)&d_imgProd, N * N * sizeof(float));
	cudaMalloc((void **)&d_realImg2, N * N * sizeof(float));
	cudaMalloc((void **)&d_imgReal2, N * N * sizeof(float));

	cudaMalloc((void **)&d_x,N * N * sizeof(float));
	cudaMalloc((void **)&ddft_r, N * N * sizeof(float));
	cudaMalloc((void **)&ddft_i, N * N * sizeof(float));

	cudaMalloc((void **)&d_sig2, N * N * sizeof(float));		
	cudaMalloc((void **)&ddft2_r, N * N * sizeof(float));
	cudaMalloc((void **)&ddft2_i, N * N * sizeof(float));

	// Generating the input matrix 1
	inputKernel<<<numberOfBlocks, numberOfThreads,0, d_x_Stream >>>(d_x, n, N);	
	cudaMemcpy(x, d_x, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	//Generating the input matrix 2
	inputKernel2<<<numberOfBlocks, numberOfThreads, 0, d_sig2_Stream >>>(d_sig2, n, N);
	cudaMemcpy(sig2, d_sig2, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("********************\n");
	printf("Input1\n");
	for(i = 0; i < n; i++)
	{
		printf("%0.3f\n",x[i][0]); 
	}
	printf("*******************\n");

	printf("Input2\n");
	for(i = 0; i < n; i++)
	{
		printf("%0.3f\n",sig2[i][0]);
	}

	// Calculating twiddle factor
	factorKernel<<<(N+511)/512, 512,0, d_w_Stream >>>(d_w, (float)N);

	cudaStreamSynchronize(d_w_Stream);

	// Calculating twiddle real matrix
	twiddleRealKernel<<<(N+511)/512, 512,0, dw_r_Stream >>>(dw_r, d_w, N);
	
	// Calculating twiddle imaginary matrix
	twiddleImgKernel<<<(N+511)/512, 512,0,dw_i_Stream >>>(dw_i, d_w, N);

	dim3 numberOfThreads_1(TILE_WIDTH, TILE_WIDTH);
	dim3 numberOfBlocks_1( (N + TILE_WIDTH -1)/TILE_WIDTH, (N + TILE_WIDTH -1)/TILE_WIDTH );

	cudaStreamSynchronize(dw_r_Stream);
	cudaStreamSynchronize(d_x_Stream);
	cudaStreamSynchronize(d_sig2_Stream);

	// Calculating real part of DFT of input matrix 1
	multKernel<<<numberOfBlocks_1, numberOfThreads_1,0,ddft_r_Stream >>>(dw_r, d_x, ddft_r, N);

	// Calculating real part of DFT of input matrix 2
	multKernel<<<numberOfBlocks_1, numberOfThreads_1, 0, ddft2_r_Stream >>>(dw_r, d_sig2, ddft2_r, N);

	cudaStreamSynchronize(dw_i_Stream);

	// Calculating imagine part of DFT of input matrix 1
	multKernel<<<numberOfBlocks_1, numberOfThreads_1,0,ddft_i_Stream >>>(dw_i, d_x, ddft_i, N);

	// Calculating imagine part of DFT of input matrix 2
	multKernel<<<numberOfBlocks_1, numberOfThreads_1,0,ddft2_i_Stream >>>(dw_i, d_sig2, ddft2_i, N);

	cudaStreamSynchronize(ddft_r_Stream);
	cudaStreamSynchronize(ddft2_r_Stream);

	//Multiplying the real part of two signals
	dotProdKernel<<<(N + 511)/512, 512,0,d_realProd_Stream >>>(ddft_r, ddft2_r, d_realProd, N);

	cudaStreamSynchronize(ddft_i_Stream);
	cudaStreamSynchronize(ddft2_i_Stream);
	
	//Multiplying the imaginary part of the two signals
	dotProdKernel<<<(N + 511)/512, 512,0,d_imgProd_Stream >>>(ddft_i, ddft2_i, d_imgProd, N);

	//Multiplying the real part of 1 and imaginary part of 2
	dotProdKernel<<<(N + 511)/512, 512,0,d_realImg2_Stream >>>(ddft_r, ddft2_i, d_realImg2, N);

	//Multiplying the img part of 1 and real part of 2
	dotProdKernel<<<(N + 511)/512, 512, 0, d_imgReal2_Stream >>>(ddft_i, ddft2_r, d_imgReal2, N);

	cudaStreamSynchronize(d_realProd_Stream);
	cudaStreamSynchronize(d_imgProd_Stream);

	// Calculating twiddle imaginary matrix for IDFT
	twiddleImgKernelIDFT<<<(N+511)/512, 512,0,dw_i2_Stream >>>(dw_i, d_w, N);

	//Real Part of DFT of Result
	subMat<<<(N*N + 511)/512, 512, 0, ddft_realRes_Stream >>>(d_realProd, d_imgProd, ddft_realRes, N);
	
	cudaStreamSynchronize(d_imgReal2_Stream);
	cudaStreamSynchronize(d_realImg2_Stream);

	//Img Part of DFT of Result
	addMat<<<(N*N + 511)/512, 512, 0, ddft_imgRes_Stream >>>(d_imgReal2, d_realImg2, ddft_imgRes, N);

	cudaStreamSynchronize(ddft_realRes_Stream);
	cudaStreamSynchronize(ddft_imgRes_Stream);

	//Real Part of Resultant Signal after taking IDFT = Real Part of Convolution
	multKernel<<<numberOfBlocks_1, numberOfThreads_1, 0, d_realRes1_Stream >>>(dw_r, ddft_realRes, d_realRes1, N);
	multKernel<<<numberOfBlocks_1, numberOfThreads_1, 0, d_imgRes2_Stream >>>(dw_r, ddft_imgRes, d_imgRes2, N);

	cudaStreamSynchronize(dw_i2_Stream);	

	//Img Part of Resultant Signal after taking IDFT = Img Part of Convolution
	multKernel<<<numberOfBlocks_1, numberOfThreads_1, 0,d_realRes2_Stream >>>(dw_i, ddft_imgRes, d_realRes2, N);
	multKernel<<<numberOfBlocks_1, numberOfThreads_1, 0, d_imgRes1_Stream >>>(dw_i, ddft_realRes, d_imgRes1, N);
	
	cudaStreamSynchronize(d_realRes1_Stream);
	cudaStreamSynchronize(d_realRes2_Stream);

	subMat<<<(N*N +511)/512, 512, 0,d_realRes_Stream >>>(d_realRes1, d_realRes2, d_realRes, N);

	cudaStreamSynchronize(d_realRes_Stream);

	divMat<<<(N*N + 511)/512, 512>>>(d_realRes, N);
	cudaMemcpy(realRes, d_realRes, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	printf(“Final Convolution\n”);
	for(i = 0; i < (2*n - 1); i++)
	{
		printf("%0.3f\n", realRes[i][0]);
	}

	cudaStreamSynchronize(d_imgRes1_Stream);
	cudaStreamSynchronize(d_imgRes2_Stream);
	
	addMat<<<(N*N + 511)/512, 512, 0, d_imgRes_Stream >>>(d_imgRes1, d_imgRes2, d_imgRes, N);

	cudaStreamSynchronize(d_imgRes_Stream);

	divMat<<<(N*N + 511)/512, 512>>>(d_imgRes, N);
	cudaMemcpy(imgRes, d_imgRes, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("*********************************************************************************\n");

	/* Timer */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time taken : %3.1f ms\n", elapsedTime);

	/* De-allocating memory on device */
	cudaFree(ddft_realRes);
	cudaFree(ddft_imgRes);
	cudaFree(d_realRes1);
	cudaFree(d_realRes2);
	cudaFree(d_realRes);
	cudaFree(d_imgRes1);
	cudaFree(d_imgRes2);
	cudaFree(d_imgRes);
	cudaFree(d_w);
	cudaFree(dw_r);
	cudaFree(dw_i);
	cudaFree(d_realProd);
	cudaFree(d_imgProd);
	cudaFree(d_realImg2);
	cudaFree(d_imgReal2);
	cudaFree(d_x);
	cudaFree(ddft_r);
	cudaFree(ddft_i);
	cudaFree(d_sig2);
	cudaFree(ddft2_r);
	cudaFree(ddft2_i);

	return 0;

}
