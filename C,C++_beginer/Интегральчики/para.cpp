#include"Header.h"
double I1, T1, xi,xi_1;

void integralParab(int N, double a, double b) {
	for (int i = 1; i <= N; i++) {
		xi_1 = (a + (i-1) * (h));
		xi = (a + i * (h));
		T1 = T1 + (Func(xi_1) + 4 * Func((xi_1 + xi) / 2) + Func(xi));
	}
	I1 = h / 6 * T1;
	T1 = 0;
}