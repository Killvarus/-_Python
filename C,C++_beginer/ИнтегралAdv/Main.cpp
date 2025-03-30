#include"Header.h"
int main()
{
	setlocale(LC_ALL, "Russian");
	double toch,a,b;
	int n;
	extern int kol, N;
	cout << "Введите значения левой границы"<<endl;
	cin >> a;
	cout << "Введите значение правой границы" << endl;
	cin>>b;
	cout << "Введите количество отрезков начальное количество отрезков разбиения"<<endl;
	cin >> n;
	cout << "Введите точность ответа"<<endl;
	cin >> toch;
	printf("Левые прямоугольники: %.6lf \n", Tochnost(sin_2, lrectangle, n, a, b, toch));
	cout << "Количество отрезков разбиения: " << N << endl;
	cout << "Количество удвоений: " << kol << endl<<endl;

	printf("Правые прямоугольники: %.6lf \n", Tochnost(sin_2, rrectangle, n, a, b, toch));
	cout << "Количество отрезков разбиения: " << N << endl;
	cout << "Количество удвоений: " << kol << endl<<endl;

	printf("Средние прямоугольники: %.6lf \n", Tochnost(sin_2, mrectangle, n, a, b, toch));
	cout << "Количество отрезков разбиения: " << N << endl;
	cout << "Количество удвоений: " << kol << endl << endl;

	printf("Трапеции: %.6lf \n", Tochnost(sin_2, Trap, n, a, b, toch));
	cout << "Количество отрезков разбиения: " << N << endl;
	cout << "Количество удвоений: " << kol << endl << endl;

	printf("Параболы: %.6lf \n", Tochnost(sin_2, Parab, n, a, b, toch));
	cout << "Количество отрезков разбиения: " << N << endl;
	cout << "Количество удвоений: " << kol << endl << endl;
	




}