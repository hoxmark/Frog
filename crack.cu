#include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include "cuda.h"
#include <cuda.h>

#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

// // char **device_hashes;
// // char **host_hashes;
// // char **help_array;
int num_lines = 1000000;

int line_length = 64;

char *host_hashes;
char *device_hashes;

const int N = 16;
const int blocksize = 16;
__device__ char *target = "93eb2df432f7d1b7281568260bbf03e06bc0b5b344ea41a1bd2ac440f5655a0f";
__device__ int d_answer = 0;

// // int **devicePointersStoredInDeviceMemory;

// void device_allocation();

__global__ void hello(char *hashes)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    bool found = true;
    int i;
    char hash[64];
    for (i = id * 64; i < (id + 1) * 64; i++)
    {
        hash[i-id*64] = hashes[i];
        if (hashes[i] != target[i - id * 64])
        {
            found = false;
        }
    }

    // printf("%d: \t %s \n", id, hash);

    if (found)
    {
        printf("We found it %d \n", id);
        d_answer = 1;
    }
}

int main()
{
    printf("1\n");
    int total_length = num_lines * line_length;
    printf("Total length %d\n", total_length);
    FILE *fp = fopen("/datadrive/crackstation_hash.txt", "r");
    host_hashes = (char *)malloc(num_lines * line_length * sizeof(char));

    if (fgets(host_hashes, total_length, fp) == NULL)
    {
        printf("We failed\n");
    }
    for (int i = 0; i < 64; i++)
    {
        printf("%c", host_hashes[i]);
    }

    printf("\n");

    cudaCheckErrors("after host buffer");
    cudaMalloc((void **)&device_hashes, total_length * sizeof(char));
    cudaMemcpy(device_hashes, host_hashes, total_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckErrors("after copy to device");


    /* WHY DO WE NEED THIS ?!?!?!?!? */
    char a[N] = "Hello \0\0\0\0\0\0";
    const int csize = N * sizeof(char);
    char *ad;
    cudaMalloc((void **)&ad, csize);
    cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);

    cudaCheckErrors("before kernel run ");
    dim3 dimBlock(1000, 1);
    dim3 dimGrid(1000, 1);

    /* Timing */

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    hello<<<dimGrid, dimBlock>>>(device_hashes);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f\n", elapsedTime);
    cudaCheckErrors("After kernel run ");

    /* WHY DO WE NEED HTIS ?!?!?! */
    cudaFree(ad);

    cudaCheckErrors("After free 1 ");
    cudaCheckErrors("After free 2 ");
    return EXIT_SUCCESS;
}
