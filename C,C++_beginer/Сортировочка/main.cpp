#include"Header.h"

int main()
{
	setlocale(LC_ALL, "Russian");
	TYPE* t,*z,*x;
	size_t N;
	unsigned int n;
	extern int pere, srav;
	printf("������� ������ ������������ �������: ");
	scanf("%u", &n);
	N = sizeof(TYPE)*n;
	t = (TYPE*)malloc(N);
	z = (TYPE*)malloc(N);;
	x = (TYPE*)malloc(N);

	/*for (int i = 0; i < n; i++)
	{
		t[i] = (float)rand() / RAND_MAX * (300 - 0) + 0;
	}
	for (int i = 0; i < n; i++)
	{
		z[i] = (float)rand() / RAND_MAX * (300 - 0) + 0;
	}
	for (int i = 0; i < n; i++)
	{
		x[i] = (float)rand() / RAND_MAX * (300 - 0) + 0;
	}
	*/
	for (int i = 0; i < n; i++)
	{
		t[i] =z[i]=x[i]= (float)rand() / RAND_MAX * (300 - 0) + 0;
	}
	printf("����������� ������ 1: \n");
	vivod(t, n);
	printf("����������� ������ 2: \n");
	vivod(z, n);
	printf("����������� ������ 3: \n");
	vivod(x, n);
	printf("������ 1, ��������������� ������� �������: \n");
	Vstavka(t, n);
	vivod(t, n); 
	printf("���������� ���������: %d\t���������� ������������: %d\n", srav, pere);
	printf("������ 2, ��������������� ������� ��������: \n");
	Pyzir(z, n);
	vivod(z, n);
	printf("���������� ���������: %d\t���������� ������������: %d\n", srav, pere);
	printf("������ 3, ��������������� ������� ������: \n");
	Vibor(x, n);
	vivod(x, n);
	printf("���������� ���������: %d\t���������� ������������: %d\n", srav, pere);



	return 0;
}