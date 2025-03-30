#include"Header.h"
int main()
{
	double a, b,e;
	extern int kol;
	extern double lev, prav;
	setlocale(LC_ALL, "Russian");
	cout << "Введите значение левой границы ";
	cin >> a;
	cout << "Введите значение правой границы ";
	cin >> b;
	cout << "Введите желаемую точность ";
	cin >> e;
	/*printf("Хорды: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", hord(tg_2x, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n",tg_2x(prav),kol);
	printf("Дитохомии: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", dichotomy(tg_2x, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", tg_2x(prav), kol);
	printf("Касательные: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", kasat(tg_2x,tan_2x_pr,a,b,e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", tg_2x(prav), kol);*/
	printf("Хорды: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", hord(_exp_x, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", _exp_x(prav), kol);
	printf("Дитохомии: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", dichotomy(_exp_x, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", _exp_x(prav), kol);
	/*printf("Касательные: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", kasat(_exp_x, _exp_x_pr, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", _exp_x(prav), kol);*/
	printf("Итераций: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", iter(_exp_x1, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", _exp_x(prav), kol);
	printf("Хорды: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", hord(i_hord, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", i_hord(prav), kol);
	printf("Дитохомии: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", dichotomy(i_dih, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", i_dih(prav), kol);
	/*printf("Касательные: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", kasat(_exp_x, _exp_x_pr, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", _exp_x(prav), kol);*/
	printf("Итераций: x=%-7lf Точность:%-5.4lf Интервал [%7lf;%7lf]", iter(i_iter, a, b, e), e, lev, prav);
	printf(" Невязка: %7lf\tКоличество итераций: %4d\n", i_iter1(prav), kol);






	return 0;

}