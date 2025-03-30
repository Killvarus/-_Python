#include "Header.h"
/*int Count(FILE* input)
{
    fseek(input, 0, SEEK_SET);
    int count = 0;
    for (char c; !feof(input);)
    {
        c = getc(input);
        if (c == '\n' || c == '\t' || c == ' ') count++;
    }
    fseek(input, 0, SEEK_SET);
    return count;
}*/
unsigned int Count(FILE* fp)
{
    unsigned int count = 0;
    for (char c; c != '\n' || c != '\t' || c != ' ';count++)
    {
        c = getc(fp);
    }
    return count;
}

void add(list** head, char* data,unsigned int n) 
{
    list* tmp = (list*)malloc(sizeof(list));
    tmp->word = (char*)malloc(sizeof(char) * n);
    strcpy(data, tmp->word);
    tmp->next = (*head);
    (*head) = tmp;
}