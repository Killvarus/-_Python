#include"Header.h"
int main()
{
	setlocale(LC_ALL, "Russian");
	int i;
	printf("������� �����: ");
	scanf("%d", &i);
	printf("\n����� � �������� ������� ���������: %d", dvoich(i));


	return 0;
}