#include"Header.h"
int kol, N;
double Tochnost
(double(*pf2)(double), double(*pf1)(double (*pf2)(double), int, double, double),int n, double a, double b, double z)
{
	N = n;
	kol = 0;
	if (abs(pf1(pf2, n, a, b) - pf1(pf2,2*n, a, b)) > z)
	{	
		N = 2 * n;
		for(kol = 1; abs(pf1(pf2, n, a, b) - pf1(pf2, N, a, b))>z; kol++)
		{
			n = N;
			N *= 2;
		}

	}
	return pf1(pf2,N,a,b);

};
