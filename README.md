parallel-prog
=============

CUDA Projects
A) ConvDFT
This code is aimed to perform Convolution operation on two input signals in parallel. 
This is achieved using DFT on the signals. The concept used is 
* Taking DFT with twiddle factors of the input signals
* Multiplying the two DFT's 
* Taking Inverse DFT of the final result to get the convolution of the original input signals

Results have been verified using MATLAB.

B) Bucket Sort 
This program sorts an input array by bucket sort.
 * Each bucket in turn is sorted using Parallel Bubble sort
 * The array consists of float numbers, all less than 1. 
 * To find the destination bucket the float number is multiplied by 10 to get the first digit, which determines the bucket number
 
C) Histogram
This program implements the histogram functionality efficiently.
 * In this implementation, each block maintains a table in which each thread keeps
 * count for each BIN_COUNT individually.
 * Threads then calculate the sum of each BIN_COUNT and store it in a shared array, of size BIN-COUNT,
 * holding block's output. 
 * Each BIN_COUNT's sum is atomically added to the final output array holding final count
 * of each bin.
 * Restriction - BIN_COUNT max can be 64, due to shared memory restriction.
 
D)  Transpose
This program takes a matrix transpose using shared memory.
 * It takes care of memory coalescence as both memory read and memory write are
 * coalesced by accessing in colum major.
 * It takes care of bank conflicts by padding the shared memory by 1 to get optimum performance.
 * There is no thread divergence in the program.
 * Each thread works on multiple elements, which in this case = 4. 
 * Restriction - Rows and Columns should be multiple of Tile Dimension
 
E) Convolution
This program takes the convolution of a given matrix by running the convolution filter
 * The filter is of size 3x3 and is hardcoded.
 * The program effectively takes care of padding.
 
F) OddInt
This program finds the count of Odd numbers in an input integer array.
 * The program uses shared memory to count the occurence of odd number in each block. 
 * The shared memory counter is then added using parallel reduction algorithm.
 * Bank conflicts are avoided using padding in the shared memory.
 * Output of each block is passed back to the CPU, where all of them are added to get the final count.
 
© 2014. All rights reserved. Property of Anand Goyal. Generated for reference educational purposes only.
