#include <stdio.h>
#include <cuda.h>
#include "sha256.cu"

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

int num_lines = 1000000;
int line_length = 64;
char **host_passwords; 
char **device_hashes;
__device__ char *target = "93eb2df432f7d1b7281568260bbf03e06bc0b5b344ea41a1bd2ac440f5655a0f";
__device__ int d_answer = 0;

__global__ void compare_hashes(char *hashes)
{
    printf("Inside kernel \n");
}

int main()
{

    int lines_allocated = 1000000;

    /* Max line len in characters*/
    int max_line_len = 40;

    /* Allocate lines of text */
    host_passwords = (char **)malloc(sizeof(char *) * lines_allocated);
    if (host_passwords == NULL)
    {
        fprintf(stderr, "Out of memory (1).\n");
        exit(1);
    }

    FILE *fp = fopen("10_million_password_list_top_1000000.txt", "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file.\n");
        exit(2);
    }

    int i;
    for (i = 0; 1; i++)
    {
        int j;


        /* Have we gone over our line allocation? */
        if (i >= lines_allocated)
        {
            printf("We have gone over our line allocation\n");
            int new_size;

            /* Double our allocation and re-allocate */
            new_size = lines_allocated * 2;
            host_passwords = (char **)realloc(host_passwords, sizeof(char *) * new_size);
            if (host_passwords == NULL)
            {
                fprintf(stderr, "Out of memory.\n");
                exit(3);
            }
            lines_allocated = new_size;
        }
        /* Allocate space for the next line */

        host_passwords[i] = (char *)malloc(max_line_len);

        if (host_passwords[i] == NULL)
        {
            fprintf(stderr, "Out of memory (3).\n");
            exit(4);
        }
        if (fgets(host_passwords[i], max_line_len, fp) == NULL)
        {
            printf("we are breaking!\n");
            break;
        }

        /* Get rid of CR or LF at end of line */
        for (j = strlen(host_passwords[i]) - 1; j >= 0 && (host_passwords[i][j] == '\n' || host_passwords[i][j] == '\r'); j--)
            ;
        host_passwords[i][j + 1] = '\0';
    }
    /* Close file */
    fclose(fp);

 
    for (int j = 0; j < 10; j++)
    {
        printf("%s\n", host_passwords[j]);
    }


    cudaMalloc((void***)&device_hashes,  num_lines*sizeof(char*));
    for(i=0; i<1; i++) {
        printf("%d\n", strlen(host_passwords[i]));
        cudaMalloc((void**) &(device_hashes[i]), strlen(host_passwords[i])*sizeof(char));
        cudaMemcpy(device_hashes[i], host_passwords[i],  strlen(host_passwords[i]) * sizeof(char), cudaMemcpyHostToDevice);
    }
        
    // dim3 dimBlock(1, 1);
    // dim3 dimGrid(1, 1);

    // /* Timing */
    // cudaEvent_t start, stop;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventRecord(start, 0);
    // compare_hashes<<<dimGrid, dimBlock>>>(device_hashes);

    // cudaEventCreate(&stop);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("Elapsed time: %f\n", elapsedTime);
    // cudaCheckErrors("After kernel run ");

    // cudaCheckErrors("After free 1 ");

    return EXIT_SUCCESS;
}
