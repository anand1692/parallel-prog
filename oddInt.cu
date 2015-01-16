/* This program finds the count of Odd numbers in an input integer array.
 * The program uses shared memory to count the occurence of odd number in each block. 
 * The shared memory counter is then added using parallel reduction algorithm.
 * Bank conflicts are avoided using padding in the shared memory.
 * Output of each block is passed back to the CPU, where all of them are added to get the final count.
 * Implemented in CUDA
 *
 *
 *
 * coded by Anand Goyal. Dated: 12/13/2014
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define SIZE 2000000
#define THREAD_NUM 4

__global__ void countOddKernel(int *inData, long size, int *count)
{
	__shared__ int temp[THREAD_NUM + 1];		

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size) {
		if(inData[tid] % 2 != 0)
			temp[threadIdx.x] = 1;
		else
			temp[threadIdx.x] = 0;
	}
	__syncthreads();

	int i = blockDim.x/2;
	while(i != 0) {
		if(threadIdx.x < i)
			temp[threadIdx.x] += temp[threadIdx.x + i];

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0)
		count[blockIdx.x] = temp[0];
}

int main()
{
	int *data, *dev_data, *count, *dev_c;
	int i, total_count = 0;
	int numOfBlocks = (SIZE + THREAD_NUM - 1)/THREAD_NUM;	
	float elapsedTime;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);	

	data = (int *)malloc(sizeof(int) * SIZE);
	count = (int *)malloc(sizeof(int) * numOfBlocks );
	cudaMalloc((void **)&dev_data, sizeof(int) * SIZE);
	cudaMalloc((void **)&dev_c, sizeof(int) * numOfBlocks);

	srand(time(NULL));
	for(i = 0; i < SIZE; i++) {
		data[i] = rand()%100 + 1;
	}
	
/*	for(i = 0; i < SIZE; i++) 
		printf("%d\n", data[i]);

	printf("*************************\n");
*/
	cudaEventRecord(start, 0);

	cudaMemcpy(dev_data, data, sizeof(int) * SIZE, cudaMemcpyHostToDevice);	
	countOddKernel<<<numOfBlocks, THREAD_NUM>>>(dev_data, SIZE, dev_c);
	cudaMemcpy(count, dev_c, sizeof(int) * numOfBlocks, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	for(i = 0; i < numOfBlocks; i++) {
		total_count += count[i];
	}

//	printf("Number of Odd numbers = %d\n", total_count);
	printf("Time :  %3.1f ms \n", elapsedTime);
	
	free(data);
	free(count);
	cudaFree(dev_data);
	cudaFree(dev_c);

	return 0;
}
