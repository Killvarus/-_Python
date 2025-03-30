#include"Header.h"
double mrectangle
(double (*pf)(double),int n, double a, double b)
{
	double s = 0;
	for (int i = 0; i < n; i++)
	{
		s += pf(a + h / 2 + i * h);
	}
	return h * s;
}
double Parab
(double(*pf)(double), int n, double a, double b) 
{
	double xi_1,xi,s=0;
		for (int i = 1; i <= n; i++)
		{
		
		xi_1 = (a + (i - 1) * (h));
		xi = (a + i * (h));
		s +=(pf(xi_1) + 4 * pf((xi_1 + xi) / 2) + pf(xi));
		}
		return(h / 6 * s);
}
double Trap
(double(*pf)(double), int n, double a, double b)
{
	double x,s=0;
	for (int i = 0; i < n; i++) 
	{
		x = a + i * h;
		s += pf(x);
	}
	return( h / 2 * (pf(a) + pf(b) + 2 * s));
}
double lrectangle
(double (*pf)(double), int n, double a, double b) 
{
	double s = 0;
	for (int i = 1; i <= n; i++)
	{
		s += pf(a + (i-1) * h) * h;
	}
	return(s);
}
double rrectangle
(double (*pf)(double), int n, double a, double b)
{
	double s = 0;
	for (int i = 1; i <= n; i++)
	{
		s += pf(a + i * h) * h;
	}
	return(s);

}

