#include"Header.h"

int main()
{
	extern solv one,two;
	FILE* fp,*fp1;
	equat eq;
	solv so;
	unsigned int p;
	if ((fp = fopen("f1.txt", "r")) != NULL)
	{
		p = numberCount(fp)+1;
		eq.a = (double*)malloc((p) * sizeof(double));
		eq.b = (double*)malloc(p * sizeof(double));
		eq.c = (double*)malloc(p * sizeof(double));
		//one.imag1 = (double*)malloc(p * sizeof(double));
		//one.imag2 = (double*)malloc(p * sizeof(double));


		for (int i = 0; ; i++)
		{
			if (fscanf(fp, "%lfx^2%lfx%lf%*s", &(eq.a[i]), &(eq.b[i]), &(eq.c[i])) == EOF) break;
		}
		if ((fp1 = fopen("f2.txt", "w")) != NULL)
		{
			for (int i = 0;i<p; i++)
			{
				solve(eq.a[i], eq.b[i], eq.c[i]);
				fprintf(fp1, "x1=%lf+%lfi\n", one.real1, one.imag1);
				fprintf(fp1, "x2=%lf+%lfi\n", one.real2, one.imag2);
			}
		}
		else perror("NO f2.txt");
		
	}
	else perror("NO f1.txt");
	fclose(fp1);
	fclose(fp);
	printf("%lf%lf", eq.a[1],one.imag1);
	return 0;

}