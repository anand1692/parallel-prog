/* This program takes a matrix transpose using shared memory.
 * It takes care of memory coalescence as both memory read and memory write are
 * coalesced by accessing in colum major.
 * It takes care of bank conflicts by padding the shared memory by 1 to get optimum performance.
 * There is no thread divergence in the program.
 * Each thread works on multiple elements, which in this case = 4. 
 * Restriction - Rows and Columns should be multiple of Tile Dimension
 * Implemented in CUDA
 *
 *
 *
 * code by Anand Goyal. Dated: 12/13/2014
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define ROW 1024
#define COL 1024
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeKernel(float *inData, float *outData)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	/* Copying data into shared memory - each thread copies 4 elements : read & write coalesced */
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = inData[(y+j) * width + x];

	__syncthreads();

	/* x,y modified according to the new transposed matrix */
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	/* Copying data to output array - each thread copies 4 elemets : read & write coalesced */
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		outData[(y+j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main()
{
	int numX = ROW, numY = COL;
  	int size = numX * numY * sizeof(float);
	float *input, *output;
	float *d_input, *d_output;
	int i, j;
	float elapsedTime;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);	
  	dim3 numOfBlocks(numX/TILE_DIM, numY/TILE_DIM);
	dim3 numOfThreads(TILE_DIM, BLOCK_ROWS);

	input = (float *)malloc(size);
  	output = (float *)malloc(size);
	
	cudaMalloc((void **)&d_input, size);
  	cudaMalloc((void **)&d_output, size);

	/* Generating the input array */
  	for (i = 0; i < numX; i++)
   		for (j = 0; j < numY; j++)
   	  		input[i*numY + j] = rand()%20 + 1;

/*	//Printing input array
	for(i = 0; i < numX; i++) {
		for(j = 0; j < numY; j++)
			printf("%0.3f\t", input[i*numY + j]);
		
		printf("\n");
	}
	printf("*******************************\n");
*/
	cudaEventRecord(start, 0);

	cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
  	transposeKernel<<<numOfBlocks, numOfThreads>>>(d_input, d_output);
	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);	

	//Printing output array
/*	for(i = 0; i < numX; i++) {
		for(j = 0; j < numY; j++)
			printf("%0.3f\t", output[i*numY + j]);
		
		printf("\n");
	}
*/
	printf("Time :  %3.1f ms \n", elapsedTime);

	cudaFree(d_input);
	cudaFree(d_output);
	free(input);
	free(output);
	return 0;
}
