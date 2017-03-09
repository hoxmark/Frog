#include <openssl/sha.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int main()
{
    char password[] = "Doggers";
    size_t length = sizeof(password);
    unsigned char hash[SHA_DIGEST_LENGTH];
    unsigned char hash2[SHA_DIGEST_LENGTH];
    SHA1((unsigned char *)password, length, hash);
    char *fileName = "crackstation.txt";
    FILE *file = fopen(fileName, "r"); /* should check the result */
    char line[256];
    unsigned int ctr = 0;
    while (fgets(line, sizeof(line), file))
    {
        strtok(line, "\n");
        SHA1((unsigned char *)line, strlen(line) + 1, hash2);
        if (memcmp(hash, hash2, SHA_DIGEST_LENGTH) == 0)
        {
            printf("We found the password. It is %s\n", line);
            break;
        }
        ctr++;
        if (ctr % 1000000 == 0)
        {
            printf("Counter: %d\n", ctr);
        }
    }
    fclose(file);
    return 0;
}