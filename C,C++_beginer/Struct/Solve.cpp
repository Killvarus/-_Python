#include "Header.h"
solv one;
void solve(double a, double b, double c)
{
	double D = pow(b,2) - 4 * a * c;
	if (D < 0)
	{
		one.real1 = -b / 2/a;
		one.imag1 = sqrt(4 * a * c - pow(b, 2))/2/a;
		one.real2 = one.real1;
		one.imag2 = (-sqrt(4 * a * c - pow(b, 2)) / 2 / a);

	}
	else
	{
		one.real1 = (-b + sqrt(D)) / (2 * a);
		one.real2 = (-b - sqrt(D)) / (2 * a);
		one.imag1 = one.imag2=0;
	}

}
