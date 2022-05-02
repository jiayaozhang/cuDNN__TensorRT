#include <cuda_runtime.h>

__global__ 
void vecAddKernel(float* A_d, float* B_d, float* C_d, int n);
