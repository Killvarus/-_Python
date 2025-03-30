#include<stdio.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include<stdlib.h>
typedef struct list
{
	struct list *next;
	unsigned int ch;
	char* word;

}list;
unsigned int Count(FILE* fp);