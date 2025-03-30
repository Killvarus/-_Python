#include "Header.h"
int numberCount(FILE* input) 
{
    fseek(input, 0, SEEK_SET);
    int counter = 0;

    for(int k;!feof(input);counter++) 
    {
        if (fscanf(input, "%d", &k) == NULL) break;
    }
    return counter;
}