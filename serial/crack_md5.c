
#include "md5.c"
#include <fcntl.h>
#include <stdio.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <sys/stat.h>
// #include <sys/types.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/*-------------
* CONFIGURATION
* ----TYPES---
* -- 1: MD5 --
* -- 2: SHA1 -
* -- 3: SHA256
*-------------*/
#define type 1

#define password_length 30
#if type == 1
#define hash_length 16
#endif
#if type == 2
#define hash_length 20
#endif
#if type == 3
#define hash_length 32
#endif

//  Number of lines in the dictionary file. Must correspond with
// Hashesorg
// #define num_passwords 446426204

// merged
#define num_passwords 19922147
// #define num_passwords 1000

// top 100
// #define num_passwords 999999

char* host_dictionary_file;
int* host_dictionary_word_lengths;
int* host_start_indexes;

void calculate_hash(unsigned char* pass_cleartext, unsigned char* hash,
                    int length) {

    MD5_CTX ctx;
    md5_init(&ctx);
    md5_update(&ctx, pass_cleartext, length);
    md5_final(&ctx, hash);
}

int main() {
    int i = 0;
    FILE* fp;
    if (type == 1) {
        fp = fopen("../passwords/eharmony.txt", "r");
    }

    if (type == 2) {
        fp = fopen("../passwords/unmasked.lst", "r");
    }

    if (type == 3) {
        fp = fopen("../passwords/targets.txt", "r");
    }

    const char* dictionary_file_name =
        "/datadrive/cracklist/merged/passwords_one_line.txt";
    FILE* length_file =
        fopen("/datadrive/cracklist/merged/password_lengths.txt", "r");

    // Calculate file size
    int fd = open(dictionary_file_name, O_RDONLY);
    size_t password_file_size;
    password_file_size = lseek(fd, 0, SEEK_END);
    printf("Total file size of the dictionary file: %zu \n",
           password_file_size);

    host_dictionary_word_lengths = (int*)malloc(num_passwords * sizeof(int));
    host_start_indexes = (int*)malloc(num_passwords * sizeof(int));

    host_dictionary_file =
        (char*)mmap(0, password_file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Copy all the lengths into host_dictionary_word_lengths
    i = 0;
    int counter = 0;
    int start = 0;
    fscanf(length_file, "%d", &i);
    host_dictionary_word_lengths[counter] = i;
    host_start_indexes[counter] = 0;
    start += i;
    counter++;
    while (!feof(length_file) && counter < num_passwords) {
        fscanf(length_file, "%d", &i);
        host_dictionary_word_lengths[counter] = i;
        host_start_indexes[counter] = start;
        start += i;
        counter++;
    }
    fclose(length_file);

    int host_num_targets = 0;
    int ch = 0;
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') {
            host_num_targets++;
        }
    }

    // host_num_targets = 100000;
    unsigned char* host_targets = (unsigned char*)malloc(
        hash_length * host_num_targets * sizeof(unsigned char));

    char* pos;
    char str[65];

    fseek(fp, 0, SEEK_SET);
    for (i = 0; i < host_num_targets; i++) {
        if (fgets(str, 100, fp) != NULL) {
            pos = str;
            int count;
            for (count = 0; count < hash_length; count++) {
                sscanf(pos, "%2hhx", &host_targets[i * hash_length + count]);
                pos += 2;
            }
        }
    }

    int k, j;
    int count = 0;

    host_num_targets = 500;
    printf("Starting computation. Hashing %d passwords, checking against %d "
           "targets \n",
           num_passwords, host_num_targets);
    clock_t begin = clock();
    for (i = 0; i < num_passwords; i++) {
        int length = host_dictionary_word_lengths[i];
        int start = host_start_indexes[i];

        if (start < password_file_size) {
            unsigned char pass_cleartext[password_length];
            if (length < password_length) {
                memcpy(pass_cleartext, &host_dictionary_file[start], length);
                pass_cleartext[length] = '\0';

                unsigned char hash[hash_length];
                calculate_hash(pass_cleartext, hash, length);

                for (k = 0; k < host_num_targets; k++) {
                    bool found = true;
                    for (j = 0; j < hash_length; j++) {
                        if (hash[j] != host_targets[k * hash_length + j]) {
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        printf(" %s \n", pass_cleartext);
                    }
                }
            }
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("We found %d passwords\n in %f time!", count, time_spent);

    return 0;
}
