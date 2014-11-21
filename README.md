parallel-prog
=============

CUDA Projects
This code is aimed to perform Convolution operation on two input signals in parallel. This is achieved using DFT on the signals. The concept used is 
1. Taking DFT with twiddle factors of the input signals
2. Multiplying the two DFT's 
3. Taking Inverse DFT of the final result to get the convolution of the original input signals
