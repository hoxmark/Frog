#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

/* Functions to be implemented: */
void device_allocation();
void ftcs_kernel();
void print_gpu_info();

int n_cores = 4992;

void print_gpu_info()
{
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    printf("Number of CUDA devices: %d\n", n_devices);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    printf("CUDA device name: %s\n", device_prop.name);
    printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
}

int main(int argc, char **argv)
{
    print_gpu_info();

    {
        int lines_allocated = 999999;
        int max_line_len = 64;

        /* Allocate lines of text */
        char **words = (char **)malloc(sizeof(char *) * lines_allocated);
        if (words == NULL)
        {
            fprintf(stderr, "Out of memory (1).\n");
            exit(1);
        }

        FILE *fp = fopen("crackstation_hash.txt", "r");
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
                words = (char **)realloc(words, sizeof(char *) * new_size);
                if (words == NULL)
                {
                    fprintf(stderr, "Out of memory.\n");
                    exit(3);
                }
                lines_allocated = new_size;
            }
            /* Allocate space for the next line */

            // Sizeof(char)?? 
            words[i] = malloc(max_line_len);
            
            if (words[i] == NULL)
            {
                fprintf(stderr, "Out of memory (3).\n");
                exit(4);
            }
            if (fgets(words[i], max_line_len - 1, fp) == NULL)
                break;

            /* Get rid of CR or LF at end of line */
            for (j = strlen(words[i]) - 1; j >= 0 && (words[i][j] == '\n' || words[i][j] == '\r'); j--)
                ;
            words[i][j + 1] = '\0';
        }
        /* Close file */
        fclose(fp);

        int j;
        for (j = 0; j < i; j++)
            printf("%s\n", words[j]);

        /* Good practice to free memory */
        for (; i >= 0; i--)
            free(words[i]);
        free(words);
        return 0;

        exit(EXIT_SUCCESS);
    }
}