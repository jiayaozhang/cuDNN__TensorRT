#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#define M 512
#define K 512
#define N 512

#define BLOCK_SIZE 32  //block size ,each thread to calucate each bloc

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

void printMatrix(float *array, int row, int col)
{
	float *p = array;
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			printf("%10lf", p[x]);
		}
		p = p + col;
		printf("\n");
	}
	return;
}


void  multiplicateMatrixOnHost(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	for (int i = 0; i < M_p; i++)
	{
		for (int j = 0; j < N_p; j++)
		{
			float sum = 0;
			for (int k = 0; k < K_p; k++)
			{
				sum += array_A[i*K_p + k] * array_B[k*N_p + j];
			}
			array_C[i*N_p + j] = sum;
		}
	}

}

__global__ void multiplicateMatrixOnDevice(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number

	if (ix < N_p && iy < M_p)
	{
		float sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy*K_p + k] * array_B[k*N_p + ix];
		}
		array_C[iy*N_p + ix] = sum;
	}
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP

	__shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;


	float Csub = 0.0;

	for (int i = 0; i < (int)(ceil((float)numAColumns / BLOCK_SIZE)); i++)
	{
		//printf("block.x=%d,block.y=%d,threadIdx.x=%d,threadIdx.y=%d,row=%d,col=%d,sharedM[%d][%d]=A[%d],A的值：%f,sharedN[%d][%d]=B[%d],B的值：%f\n",
		//	blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col,
		//	threadIdx.y, threadIdx.x, row*numAColumns + i * BLOCK_SIZE + tx, A[row*numAColumns + i * BLOCK_SIZE + tx],
		//	threadIdx.y, threadIdx.x, (i*BLOCK_SIZE + ty)*numBColumns + col, B[(i*BLOCK_SIZE + ty)*numBColumns + col]);

		if (i*BLOCK_SIZE + tx < numAColumns && row < numARows)
			sharedM[ty][tx] = A[row*numAColumns + i * BLOCK_SIZE + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*BLOCK_SIZE + ty < numBRows && col < numBColumns)
			sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();


		for (int j = 0; j < BLOCK_SIZE; j++)
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}


	if (row < numCRows && col < numCColumns)
		C[row*numCColumns + col] = Csub;

}


int main(int argc, char **argv)
{
	clock_t start = 0, finish = 0;
	float time;

	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;


	float *h_A, *h_B, *hostRef, *deviceRef;
	h_A = (float*)malloc(Axy * sizeof(float));
	h_B = (float*)malloc(Bxy * sizeof(float));

	int nBytes = M * N * sizeof(float);
	hostRef = (float*)malloc(Cxy * sizeof(float));
	deviceRef = (float*)malloc(Cxy * sizeof(float));

	initial(h_A, Axy);
	//printf("\n");
	//printf("Matrix_A: (%d×%d)\n", M, K);
	//printMatrix(h_A, M, K);
	initial(h_B, Bxy);
	//printf("Matrix_B: (%d×%d)\n", K, N);
	//printMatrix(h_B, K, N);

	start = clock();
	multiplicateMatrixOnHost(h_A, h_B, hostRef, M, K, N);
	finish = clock();
	time = (float)(finish - start) / CLOCKS_PER_SEC;

	printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnHost \n");
	printf("------------------------------------------------------------------------------------\n");

	printf("Matrix_hostRef: (%d×%d)  CPU运行时间为：%lfs\n", M, N, time);
	//printMatrix(hostRef, M, N);

	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, Axy * sizeof(float));
	cudaMalloc((void**)&d_B, Bxy * sizeof(float));
	cudaMalloc((void**)&d_C, Cxy * sizeof(float));

	cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);


	printf("\n\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnDevice \n");
	printf("------------------------------------------------------------------------------------\n");

        int dimx = 2;
        int dimy = 2;
        dim3 block(dimx, dimy);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        //      dim3 grid(1, 1);

        cudaEvent_t gpustart, gpustop;
        float elapsedTime = 0.0;
        cudaEventCreate(&gpustart);
        cudaEventCreate(&gpustop);
        cudaEventRecord(gpustart, 0);
        multiplicateMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        cudaEventRecord(gpustop, 0);
        cudaEventSynchronize(gpustop);

        cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
        cudaEventDestroy(gpustart);
        cudaEventDestroy(gpustop);


        cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
                M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);
        //printMatrix(deviceRef, M, N);


	elapsedTime = 0.0;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	matrixMultiplyShared << < grid, block >> > (d_A, d_B, d_C, M, K, K, N, M, N);
	//	printf("   multiplicateMatrixOnDevice<<<(%d,%d),(%d,%d)>>>", grid.x, grid.y, block.x, block.y);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);


	cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
		M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);
	//printMatrix(deviceRef, M, N);
/*
        elapsedTime = 0.0;
        cudaEventCreate(&gpustart);
        cudaEventCreate(&gpustop);
        cudaEventRecord(gpustart, 0);
*/
        cublasStatus_t status;
        cublasHandle_t handle;
        cublasCreate(&handle);

        elapsedTime = 0.0;
        cudaEventCreate(&gpustart);
        cudaEventCreate(&gpustop);
        cudaEventRecord(gpustart, 0);

        float a = 1, b = 0;
        cublasSgemm(
          handle,
          CUBLAS_OP_T,   //矩阵A的属性参数，转置，按行优先
          CUBLAS_OP_T,   //矩阵B的属性参数，转置，按行优先
          M,          //矩阵A、C的行数
          N,          //矩阵B、C的列数
          K,          //A的列数，B的行数，此处也可为B_ROW,一样的
          &a,             //alpha的值
          d_A,            //左矩阵，为A
          K,          //A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
          d_B,            //右矩阵，为B
          N,          //B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
          &b,             //beta的值
          d_C,            //结果矩阵C
          M           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );
        cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaEventRecord(gpustop, 0);
        cudaEventSynchronize(gpustop);

        cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
        cudaEventDestroy(gpustart);
        cudaEventDestroy(gpustop);

        printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
                M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(deviceRef);

	cudaDeviceReset();

	return (0);
}

