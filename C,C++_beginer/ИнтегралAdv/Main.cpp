#include"Header.h"
int main()
{
	setlocale(LC_ALL, "Russian");
	double toch,a,b;
	int n;
	extern int kol, N;
	cout << "������� �������� ����� �������"<<endl;
	cin >> a;
	cout << "������� �������� ������ �������" << endl;
	cin>>b;
	cout << "������� ���������� �������� ��������� ���������� �������� ���������"<<endl;
	cin >> n;
	cout << "������� �������� ������"<<endl;
	cin >> toch;
	printf("����� ��������������: %.6lf \n", Tochnost(sin_2, lrectangle, n, a, b, toch));
	cout << "���������� �������� ���������: " << N << endl;
	cout << "���������� ��������: " << kol << endl<<endl;

	printf("������ ��������������: %.6lf \n", Tochnost(sin_2, rrectangle, n, a, b, toch));
	cout << "���������� �������� ���������: " << N << endl;
	cout << "���������� ��������: " << kol << endl<<endl;

	printf("������� ��������������: %.6lf \n", Tochnost(sin_2, mrectangle, n, a, b, toch));
	cout << "���������� �������� ���������: " << N << endl;
	cout << "���������� ��������: " << kol << endl << endl;

	printf("��������: %.6lf \n", Tochnost(sin_2, Trap, n, a, b, toch));
	cout << "���������� �������� ���������: " << N << endl;
	cout << "���������� ��������: " << kol << endl << endl;

	printf("��������: %.6lf \n", Tochnost(sin_2, Parab, n, a, b, toch));
	cout << "���������� �������� ���������: " << N << endl;
	cout << "���������� ��������: " << kol << endl << endl;
	




}