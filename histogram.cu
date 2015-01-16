/* This program implements the histogram functionality efficiently.
 * In this implementation, each block maintains a table in which each thread keeps
 * count for each BIN_COUNT individually.
 * Threads then calculate the sum of each BIN_COUNT and store it in a shared array, of size BIN_COUNT,
 * holding block's output. 
 * Each BIN_COUNT's sum is atomically added to the final output array holding final count
 * of each bin.
 * The maximum performance was observed for number of blocks as 8 x multiprocessors on GPU and number
 * of threads as the BIN_COUNT. 
 * Restriction - BIN_COUNT max can be 64, due to shared memory restriction.
 *
 *
 * 
 * code by Anand Goyal. Dated : 12/13/2014
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define SIZE 10000000
#define BIN_COUNT 64

__global__ void histKernel(char *inData, long size, unsigned int *histo)
{
	__shared__ unsigned int temp[BIN_COUNT][BIN_COUNT];
	__shared__ unsigned int blockSum[BIN_COUNT];
	int i = 0;

	while(i < BIN_COUNT) 
		temp[i++][threadIdx.x] = 0;

	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while(tid < size) {
		temp[(int)inData[tid]][threadIdx.x]++;
		tid += offset;
	}

	__syncthreads();

	i = 0;
	while(i < BIN_COUNT)
		blockSum[threadIdx.x] += temp[threadIdx.x][i++];

	__syncthreads();

	atomicAdd(&(histo[threadIdx.x]), blockSum[threadIdx.x]);
}

int main()
{
	char *buffer, *dev_buf;
	unsigned int *hist, *dev_hist;
	int i, block; 
	long count = 0;
	float timer;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	buffer = (char *)malloc(sizeof(char) * SIZE);
	hist = (unsigned int *)malloc(sizeof(unsigned int) * BIN_COUNT);
	cudaMalloc((void **)&dev_buf, sizeof(char) * SIZE);
	cudaMalloc((void **)&dev_hist, sizeof(unsigned int) * BIN_COUNT);
	cudaMemset(dev_hist, 0, BIN_COUNT * sizeof(unsigned int));

	srand(time(NULL));
	for(i = 0; i < SIZE; i++) {
		buffer[i] = (char)(rand()%64);
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	block = prop.multiProcessorCount;

	cudaEventRecord(start, 0);

	cudaMemcpy(dev_buf, buffer, SIZE * sizeof(char), cudaMemcpyHostToDevice);
	histKernel<<<block*8, BIN_COUNT>>>(dev_buf, SIZE, dev_hist);
	cudaMemcpy(hist, dev_hist, BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);

	for(i = 0; i < BIN_COUNT; i++) {
		if(hist[i] != 0) {
//			printf("Hist[%d] = %d\n", i, hist[i]); 
			count += hist[i];
		}
	}

//	printf("Hist count = %ld\n", count);
	printf("Time to scan matrix:  %3.1f ms \n", timer);
	cudaFree(dev_buf);
	cudaFree(dev_hist);
	free(buffer);
	free(hist);
	
	return 0;
}
