/* This program implements the bucket sort algorithm.
 * It uses merge sort to sort the elements within the bucket.
 * Complexity of the code is O(nlogn). 
 * In this implementaiton, float numbers 0<n<1 are generated. A number is multiplied by 10
 * to get the first digit, which in turn decides the bucket it will go into.
 * For eg: suppose number is 0.1234, then (int)0.1234*10 = 1, thus it goes to bucket 1.
 * 
 * Each process takes N/number_of_processes numbers, where N is the size of the array. It puts the 
 * elements in respective buckets. Each process then sends its data to all other
 * processes. The processes then receive only those numbers which fall into their bucket.
 * Each process then sorts its own bucket using merge sort and then sends the sorted bucket to  
 * process 0, which gathers data from all the buckets and stores continuously into the 
 * resultant array.
 * Here number of processes = RANGE = 10.
 * Implementation in MPI.
 *
 * code by Anand Goyal. Dated: 12/13/2014
 * */

#include<stdio.h>
#include<mpi.h>
#include<time.h>
#include<sys/time.h>
#include<stdlib.h>

#define SIZE 500
#define RANGE 10

/* structure of a bucket */
struct buckets {
	float *array;
	int index;
};

typedef struct buckets buckets;

/* Function to insert an element into the bucket */
void bucketInsert(buckets *b, float num) {
	b->array[b->index] = num;
	b->index = b->index + 1;
}

/* Merge function of the Merge Sort */
void merge(float *arr, float *help, int low, int mid, int high) {
	int i;
	for(i = low; i <= high; i++)
		help[i] = arr[i];

	int help_l = low;
	int help_r = mid + 1;
	int curr = low;

	while(help_l <= mid && help_r <= high) {
		if(help[help_l] < help[help_r]) {
			arr[curr] = help[help_l];
			help_l++; curr++;
		} else {
			arr[curr] = help[help_r];
			help_r++; curr++;
		}
	}

	int rem = mid - help_l;
	for(i = 0; i <= rem; i++)
		arr[curr + i] = help[help_l + i];
}

void mergeSort(float *arr, float *helper, int low, int high) {
	if(arr == NULL)
		return;

	if(low < high) {
		int mid = (low + high)/2;
		mergeSort(arr, helper, low, mid);
		mergeSort(arr, helper, mid+1, high);		
		merge(arr, helper, low, mid, high);
	}
}

int main()
{
	int comm_sz; /* Number of processes should be equal to RANGE */
	int my_rank; /* My Process rank */

	/* Initializing MPI */
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	float *input;
	int numBuckets = RANGE;
    int	bucketLength = (SIZE + comm_sz - 1)/comm_sz;
	int i;
	double localStart, localFinish, localElapsed, elapsed;

	MPI_Barrier(MPI_COMM_WORLD);
	localStart = MPI_Wtime();

	/* Only process 0 generates the input array */
	if(my_rank == 0) {
		input = (float *)malloc(sizeof(float) * SIZE);

		srand(time(NULL));
		for(i = 0; i < SIZE; i++)
			input[i] = (float)(rand()%100000 + 1)/(float)100000;

//		for(i = 0; i < SIZE; i++)
//			printf("%0.4f, ", input[i]);
//		printf("\n");
	}

	buckets **rangeBuckets = (buckets **)malloc(sizeof(buckets *) * numBuckets);
	for(i = 0; i < numBuckets; i++) {
		rangeBuckets[i] = (buckets *)malloc(sizeof(buckets));
		/* Bucket length is 2 times the ideal case, which is N/RANGE */
		rangeBuckets[i]->array = (float *)malloc(sizeof(float) * bucketLength * 2);
		rangeBuckets[i]->index = 0;
	}

	/* Local bucket of the process to store numbers with itself as destination */
	buckets localBucket;

	/* Length is 4 times because it accepts the buckets from other processes of size 2 times also */
	float *localBucketArray = (float *)malloc(sizeof(float) * bucketLength * 4);
	localBucket.array = localBucketArray;
	localBucket.index = 0;

	/* Sending the data array to all the processes with chunk size N/num_proccess */
	MPI_Scatter(input, bucketLength, MPI_FLOAT, localBucket.array, bucketLength, MPI_FLOAT, 0, 
					MPI_COMM_WORLD);
	
	/* Each process buckets each element received respectively */
	int bucketNum;
	for(i = 0; i < bucketLength; i++) {
		bucketNum = localBucket.array[i] * 10;	
		if(bucketNum == my_rank) {
			bucketInsert(&localBucket, localBucket.array[i]);
		} else {
			/* If element belongs to itself, it appends it in its local bucket */
			bucketInsert(rangeBuckets[bucketNum], localBucket.array[i]);
		}
	}

	/* Each process sends its bucket list to all other processes */
	for(i = 0; i < comm_sz; i++) {
		if(i != my_rank)
			MPI_Send(rangeBuckets[i]->array, bucketLength*2, MPI_FLOAT, i, rangeBuckets[i]->index,
							MPI_COMM_WORLD);
	}

	MPI_Status status;
	int localIndex = localBucket.index;

	/* Each process receives its respective data */
	for(i = 0; i < comm_sz-1; i++) {
		MPI_Recv(&localBucket.array[localIndex], bucketLength*2, MPI_FLOAT, MPI_ANY_SOURCE,
					   	MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		localIndex += status.MPI_TAG;
	}
	localBucket.index = localIndex;

	/* Each process sorts its respective bucket */
	float *helper = (float *)malloc(sizeof(float *) * localBucket.index);
	mergeSort(localBucket.array, helper, 0, localBucket.index-1);

	/* Size of each bucket is gathered to calculate respective offsets in final array */
	int *bucketSizes = (int *)malloc(sizeof(int) * comm_sz);
	MPI_Gather(&localIndex, 1, MPI_INT, bucketSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Displacement of each bucket in the final array */
	int *startIndex = (int *)malloc(sizeof(int) * comm_sz);
	if(my_rank == 0) {
		startIndex[0] = 0;
		for(i = 1; i < comm_sz+1; i++)
			startIndex[i] = startIndex[i-1] + bucketSizes[i-1];
	}

	/* Data is gathered from each process and put directly at its respective offset 
	 * in the main array */
	MPI_Gatherv(localBucket.array, localBucket.index, MPI_FLOAT, input, bucketSizes, startIndex, 
					MPI_FLOAT, 0, MPI_COMM_WORLD);

	localFinish = MPI_Wtime();
	localElapsed = localFinish - localStart;
	MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Process 1 prints the final output */
	if(my_rank == 0){
//		printf("***************************\n");
//		for(i = 0; i < SIZE; i++)
//			printf("%0.4f\n", input[i]);

		printf("Elapsed Time = %e seconds\n", elapsed);
	}

	MPI_Finalize();
	return 0;
}
