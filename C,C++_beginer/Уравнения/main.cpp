#include"Header.h"
int main()
{
	double a, b,e;
	extern int kol;
	extern double lev, prav;
	setlocale(LC_ALL, "Russian");
	cout << "������� �������� ����� ������� ";
	cin >> a;
	cout << "������� �������� ������ ������� ";
	cin >> b;
	cout << "������� �������� �������� ";
	cin >> e;
	/*printf("�����: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", hord(tg_2x, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n",tg_2x(prav),kol);
	printf("���������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", dichotomy(tg_2x, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", tg_2x(prav), kol);
	printf("�����������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", kasat(tg_2x,tan_2x_pr,a,b,e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", tg_2x(prav), kol);*/
	printf("�����: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", hord(_exp_x, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", _exp_x(prav), kol);
	printf("���������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", dichotomy(_exp_x, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", _exp_x(prav), kol);
	/*printf("�����������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", kasat(_exp_x, _exp_x_pr, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", _exp_x(prav), kol);*/
	printf("��������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", iter(_exp_x1, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", _exp_x(prav), kol);
	printf("�����: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", hord(i_hord, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", i_hord(prav), kol);
	printf("���������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", dichotomy(i_dih, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", i_dih(prav), kol);
	/*printf("�����������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", kasat(_exp_x, _exp_x_pr, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", _exp_x(prav), kol);*/
	printf("��������: x=%-7lf ��������:%-5.4lf �������� [%7lf;%7lf]", iter(i_iter, a, b, e), e, lev, prav);
	printf(" �������: %7lf\t���������� ��������: %4d\n", i_iter1(prav), kol);






	return 0;

}