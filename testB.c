#include <stdio.h>
#include <string.h>
#include <openssl/sha.h>
#include <stdlib.h>

int main()
{
    unsigned char ibuf[] = "bear";
    unsigned char obuf[20];
    SHA1(ibuf, strlen(ibuf), obuf);
    char const *const fileName = "10_million_password_list_top_1000000.txt"; /* should check that argc > 1 */
    FILE *file = fopen(fileName, "r");                                       /* should check the result */
    char line[256];
    while (fgets(line, sizeof(line), file)){
        strtok(line, "\n");
        unsigned char hashedLine[20];
        SHA1(line, strlen(line), hashedLine);
        int match = 1;
        for (int i = 0; i < 20; i++){
            if (hashedLine[i] != obuf[i]){
                match = 0;
                break;
            }
        }
        if (match == 1){
            printf("password is: %s", line);  
            break;
        }
    }
    fclose(file);
    return 0;
}


        // for (int i = 0; i < 256; i++)
        // {
        //     if (line[i] == '\n')
        //     {
        //         stripedLine = malloc(sizeof(char)*i); 
        //         strncpy(stripedLine, line, i);           
        //         break;
        //     }        
        // }