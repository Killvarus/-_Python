#include"Header.h"
int numberCount(FILE* input)
{
    fseek(input, 0, SEEK_SET);
    int counter = 0;
    for(char c;!feof(input);)
    {
        c = getc(input);
        if (c == '\n' || c == '\t' || c == ' ') counter++;
    }
    fseek(input, 0, SEEK_SET);
    return counter;
}