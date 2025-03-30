#include"Header.h"

int main()
{
	char c;
	FILE* fp;
	unsigned int count[26];
	for (int i=0; i < 26; i++)
	{
		count[i] = 0;
	}
	if ((fp = fopen("f1.txt", "r")) != NULL)
	{
			
			for (int k = 97, j=0; k <= 122; k++,j++)
			{
				fseek(fp, 0, SEEK_SET);
				for (; !feof(fp);)
				{
					c = getc(fp);
					if ((int)c == k || (int)c == k - 32) count[j] = count[j] + 1;
				}
			}

	}
	else perror("NO f1.txt");
	for (int i=97, j = 0; j < 26; j++,i++)
	{
		printf("%c : %u\n", (char)i, count[j]);
	}

	return 0;
}