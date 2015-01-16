/* This program takes the convolution of a given matrix by running the convolution filter
 * The filter is of size 3x3 and is hardcoded.
 * The program effectively takes care of padding.
 * Performance for a 512x512 input matrix came best for shared memory of 16K (total 48K).
 * The output of the program is re-verfied using MATLAB and using DFT properties.
 * Implemented in CUDA
 *
 *
 * 
 * code by Anand Goyal. Dated : 12/13/2014
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define NROWS 512
#define NCOLS 512
#define FILTER_ROWS 3
#define FILTER_COLS 3

__global__ void convKernel(int *inData, int *filter, int dataCol, int dataRow, int filRowRad, int filColRad,
						   int *outData)
{
	__shared__ int padRect[16*1024];
	int i, col, row, sum = 0;

	int globalCol = threadIdx.x + blockIdx.x * blockDim.x;
	int globalRow = threadIdx.y + blockIdx.y * blockDim.y;
	int globalIdx = globalCol * dataRow + globalRow;

	int localIdx = threadIdx.x * blockDim.y + threadIdx.y;
	int localCells = blockDim.x * blockDim.y;

	int padRectCol = threadIdx.x + filColRad;
	int padRectRow = threadIdx.y + filRowRad;
	int padRectOffset = 2*filRowRad + blockDim.y;
	int padRectCells = padRectOffset * (blockDim.x + 2*filColRad);

	int *padRectOut = (int*)&padRect[((padRectCells-1)/32 + 1) * 32]; //Padding up with 32
	padRectOut[localIdx] = 0;

	int filOffset = filRowRad*2 + 1;
	int filCells = filOffset * (filColRad*2 + 1);
	int *localFilter = (int *)&padRectOut[((localCells-1)/32 + 1) * 32]; //Padding up with 32

	// Copying the filter elements to shared memory
	for(i = 0; i < (filCells/localCells) + 1; i++) {
		int index = i*localCells + localIdx;
		if(index < filCells) {
			localFilter[index] = filter[index];
		}
	}

	// Copying the Data elements to padded shared memory
	for(i = 0; i < (padRectCells/localCells) + 1; i++) {
		int index = i*localCells + localIdx;
		if(index < padRectCells) {
			int prCol = index / padRectOffset;
			int prRow = index % padRectOffset;
			int glCol = prCol + blockIdx.x*blockDim.x - filColRad;
			int glRow = prRow + blockIdx.y*blockDim.y - filRowRad;
			int glIdx = glCol * dataRow + glRow;
			if(glRow >= 0 && glRow < dataRow && glCol >= 0 && glCol < dataCol)
				padRect[index] = inData[glIdx];
			else
				padRect[index] = 0;
		}
	}

	__syncthreads();

	//Taking Convolution
	for(col = -filColRad; col <= filColRad; col++) {
		for(row = -filRowRad; row <= filRowRad; row++) {
			int filCol = filColRad - col;
			int filRow = filRowRad - row;
			int filIdx = filCol*filOffset + filRow;
			int filVal = localFilter[filIdx];

			int prCol = padRectCol + col;
			int prRow = padRectRow + row;
			int prIdx = prCol*padRectOffset + prRow;
			sum += filVal * padRect[prIdx];
		}
	}
	
	padRectOut[localIdx] = sum;
	__syncthreads();

	outData[globalIdx] = padRectOut[localIdx];


}

int main()
{
	int *input, *output, *filter;
	int *d_input, *d_output, *d_filter;
	int i, j;
	float elapsedTime;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);

	input = (int *)malloc(sizeof(int) * NROWS*NCOLS);
	output = (int *)malloc(sizeof(int) * NROWS*NCOLS);
	filter = (int *)malloc(sizeof(int) * FILTER_ROWS*FILTER_COLS);

	cudaMalloc((void **)&d_input, sizeof(int) * NROWS*NCOLS);
	cudaMalloc((void **)&d_output, sizeof(int) * NROWS*NCOLS);
	cudaMalloc((void **)&d_filter, sizeof(int) * FILTER_ROWS*FILTER_COLS);

	dim3 numOfThreads(16, 16, 1);
	dim3 numOfBlocks(NROWS/16, NCOLS/16, 1);

	//Populating Input Matrix
	for(i = 0; i < NROWS; i++)
		for(j = 0; j < NCOLS; j++)
			input[i*NCOLS + j] = rand()%20 - 10;
	
	//Populating Filter Matrix
	filter[0] = -1; filter[1] = 0; filter[2] = 1;
	filter[3] = -2; filter[4] = 0; filter[5] = 2;
	filter[6] = -1; filter[7] = 0; filter[8] = 1;

	cudaEventRecord(start, 0);
	cudaMemcpy(d_input, input, sizeof(int)*NROWS*NCOLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, sizeof(int)*FILTER_ROWS*FILTER_COLS, cudaMemcpyHostToDevice);
	convKernel<<<numOfBlocks, numOfThreads>>>(d_input, d_filter, NROWS, NCOLS, FILTER_ROWS/2, FILTER_COLS/2,
											  d_output);
	cudaMemcpy(output, d_output, sizeof(int) * NROWS*NCOLS, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);	

/*	printf("After Convolution :\n");
	for(i = 0; i < NROWS; i++) {
		for(j = 0; j < NCOLS; j++)	
			printf("%d ",output[i*NCOLS + j]);
		printf("\n");
	}
*/
	printf("Time :  %3.1f ms \n", elapsedTime);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);
	free(input);
	free(output);
	free(filter);
	return 0;
}
