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
			word = (char*)malloc(Count(fp) * sizeof(char));//������� ������ ��� ������� ����, ����� �������� ���� ������ �� �����
			fscanf(fp, "%s", word);
			for (int i = 0;!feof(fp); i++) //���� ��� ��������� ���� ������ �� ����� ������� � ����� 
			{
				rex = (list*)malloc(sizeof(list));
				if (strcmp(one.word, word) != 0) //���� ������ �� �������, �� ������� ������ ��� ��� ���� ������� �����
				{
					*(rex+i)->word = (char*)malloc(Count(fp) * sizeof(char));
					strcpy(rex->word, word);
				}
				//else �� ������
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