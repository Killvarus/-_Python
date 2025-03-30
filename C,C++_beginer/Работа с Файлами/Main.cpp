#include "Header.h"
int main()
{
	setlocale(0, "Rus");
	int* mass;
	FILE* fp, * fp1;
	int p;
	if ((fp = fopen("f2.txt", "r")) != NULL)
	{
		p = numberCount(fp);
		fseek(fp, 0, SEEK_SET);
		mass = (int*)malloc(sizeof(int) * p);
		for (int i = 0; i < p; i++)
		{
			fscanf(fp, "%d", &mass[i]);
		}
	}
	else perror("NO f2.txt");
	fp1 = fopen("f3.txt", "w");
	for (int i = 0; i < p; ++i) 
	{	
		if(mass[i]==1) fprintf(fp1,"%-6d", mass[i]);
		else 
		{
			for (int j = 2; j < sqrt(mass[i]); j++)
		{
			if (mass[i] % j == 0) break;
			else
			{
				if(j>=(int)sqrt(mass[i])) fprintf(fp1,"%-6d", mass[i]);
			}
		}
		}
		
	}
	fclose(fp1);
	fclose(fp);
	return 0;
}
	


