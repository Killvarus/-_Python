#include"Header.h"
double I,T,x,Znach1;

void integralTrap(int N, double a, int b) {
	for (int i = 0; i < N; i++) {
		x = a + i * h;
		T = T + Func(x);

	}
	I = h / 2 * (Func(a) + Func(b) + 2 * T);
	T = 0;
}