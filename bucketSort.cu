/* This program sorts an input array by bucket sort.
 * Each bucket in turn is sorted using Parallel Bubble sort.
 * The array consists of float numbers, all less than 1. To find the destination bucket,
 * the float number is multiplied by 10 to get the first digit, which determines the bucket number.
 * For eg., 0.1234 -> (int)(0.1234*10) = 1. Thus the bucket number for 0.1234 is 1.
 * Thus the total number of buckets will be 10. (0-9)
 * Implemented in CUDA. 
 *
 *
 * 
 * code by Anand Goyal. Dated: 12/13/2014
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define range 10
#define SIZE 5000
#define bucketLength (SIZE/range * 2)

__global__ void bucketSortKernel(float *inData, long size, float *outData)
{
	__shared__ float localBucket[bucketLength];
	__shared__ int localCount; /* Counter to track index with a bucket */

	int tid = threadIdx.x; int blockId = blockIdx.x;
	int offset = blockDim.x;
	int bucket, index, phase;
	float temp;
	
	if(tid == 0)
		localCount = 0;

	__syncthreads();

	/* Block traverses through the array and buckets the element accordingly */
	while(tid < size) {
		bucket = inData[tid] * 10;
		if(bucket == blockId) {
			index = atomicAdd(&localCount, 1);
			localBucket[index] = inData[tid]; 
		}
		tid += offset;		
	}

	__syncthreads();
	
	tid = threadIdx.x;
	//Sorting the bucket using Parallel Bubble Sort
	for(phase = 0; phase < bucketLength; phase ++) {
		if(phase % 2 == 0) {
			while((tid < bucketLength) && (tid % 2 == 0)) {
				if(localBucket[tid] > localBucket[tid +1]) {
					temp = localBucket[tid];
					localBucket[tid] = localBucket[tid + 1];
					localBucket[tid + 1] = temp;
				}
				tid += offset;
			}
		}
		else {
			while((tid < bucketLength - 1) && (tid %2 != 0)) {
				if(localBucket[tid] > localBucket[tid + 1]) {
					temp = localBucket[tid];
					localBucket[tid] = localBucket[tid + 1];
					localBucket[tid + 1] = temp;
				}
				tid += offset;
			}
		}
	}
	
	tid = threadIdx.x;
	while(tid < bucketLength) {
		outData[(blockIdx.x * bucketLength) + tid] = localBucket[tid];
		tid += offset;
	}
}

int main()
{
	float *input, *output;
	float *d_input, *d_output;
	int i;
	float elapsedTime;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);

	/* Each block sorts one bucket */
	const int numOfThreads = 4;
	const int numOfBlocks = range;

	input = (float *)malloc(sizeof(float) * SIZE);
	output = (float *)malloc(sizeof(float) * bucketLength * range);
	cudaMalloc((void**)&d_input, sizeof(float) * SIZE);
	cudaMalloc((void **)&d_output, sizeof(float) * bucketLength * range);
	cudaMemset(d_output, 0, sizeof(float) * bucketLength * range);

	/* Generating the input array to be sorted */
	srand(time(NULL));
	for(i = 0; i < SIZE; i++)
		input[i] = (float)(rand()%10000 + 1)/(float)10000;

	// Printing the input array
/*	for(i = 0; i < SIZE; i++)
		printf("%0.4f\n", input[i]);

	printf("***********************\n");
*/
	cudaEventRecord(start, 0);

	cudaMemcpy(d_input, input, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	bucketSortKernel<<<numOfBlocks, numOfThreads>>>(d_input, SIZE, d_output);
	cudaMemcpy(output, d_output, sizeof(float) * bucketLength * range, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

/*	//Printing the sorted array
	for(i = 0; i < range; i++) {
		for(j = 0; j < bucketLength; j++)
			if(output[i*bucketLength + j] != 0)
				printf("%0.4f ", output[i*bucketLength + j]);
	}

	printf("\n");
*/	printf("Time :  %3.1f ms \n", elapsedTime);

	cudaFree(d_input);
	cudaFree(d_output);
	free(input);
	free(output);

	return 0;
}
