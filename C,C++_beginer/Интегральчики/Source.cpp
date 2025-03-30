#include"Header.h"
extern double I,I1,Znach1;
int ud;
int main() {
	double a, b,U,K,z;
	setlocale(LC_ALL, "Russian");
	int N,N1;
	cout << "¬ведите значение левой границы: ";
	cin >> a;
	cout << "¬ведите значение правой границы: ";
	cin >> b;
	cout << "¬ведите количество отрезков разбиени€: ";
	cin >> N;
	cout << "¬ведите желаемую точность: ";
	cin >> z;	
	N1 = N;
	integralTrap(N, a, b);
	K = I;
	integralTrap(N*2, a, b);
	U = I;
	if (fabs(K - U)>z) {
		for (int i = 1; fabs(K - U) > z && i<20; i++) {
			integralTrap(2*N, a, b);
			K = I;
			integralTrap(4 * N, a, b);
			U = I;
			ud++;
			N = N*2;
		}
	}
	cout << endl << endl << "«начение интеграла, вычесленного методом трапеций: ";
	printf("%.15lf\n",I);
	cout << "ѕри этом интервал [" << a << "," << b << "] был разбит на " << N << " отрезков" << endl;
	cout << "ѕотребовалось " << ud << " удвоений количества отрезков" << endl;
	cout<< "ƒостигнута€ точность (веро€тно): " << z << endl;
	integralParab(N, a, b);
	K = I1;
	I1 = 0;
	ud = 0;
	integralParab(N * 2, a, b);
	U = I1;
	if (fabs(K - I1) > z) {
		for (int i = 1; fabs(K - U) > z && i<10; i++) {
			integralParab(2 *N1, a, b);
			K = I1;
			integralParab(4* N, a, b);
			U = I1;
			ud++;
			N1 = N1 * 2;
		}
	}
	cout << endl << endl << "«начение интеграла, вычесленного методом парабол: ";
	printf("%.15lf\n", I1);
	cout << "ѕри этом интервал [" << a << "," << b << "] был разбит на " << N1 << " отрезков" << endl;
	cout << "ѕотребовалось " << ud << " удвоений количества отрезков" << endl;
	cout << "ƒостигнута€ точность (веро€тно): " << z << endl;
	return 0;
}