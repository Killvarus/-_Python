#include"Header.h"

int main()
{
	list one;
	list* rex;
	list* beg = NULL;
	list* end = NULL;
	FILE* fp;
	char* word;
	if ((fp = fopen("f1.txt", "r")) != NULL)
	{

		for (int i = 0; !feof(fp); i++)
		{
			word = (char*)malloc(Count(fp) * sizeof(char));//Выделяю память для массива ворд, чтобы записать туда строку из файла
			fscanf(fp, "%s", word);
			for (int i = 0;!feof(fp); i++) //Цикл для сравнения этой строки со всеми прочими в листе 
			{
				rex = (list*)malloc(sizeof(list));
				if (strcmp(one.word, word) != 0) //Если строки не совпали, то выделяю память под еще один элемент листа
				{
					*(rex+i)->word = (char*)malloc(Count(fp) * sizeof(char));
					strcpy(rex->word, word);
				}
				//else не забыть
				if (beg == NULL && end == NULL)
					beg = rex;
				else
				{
					end->next = rex;
					end = rex;
					end->next = NULL;
				}
			}
			
			{


			}
		}



	}
	else perror("NO f1.txt");




}