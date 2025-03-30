#include"Header.h"
int main()
{
	setlocale(LC_ALL, "Russian");
	int i;
	printf("¬ведите число: ");
	scanf("%d", &i);
	printf("\n„исло в двочиной системе счислени€: %d", dvoich(i));


	return 0;
}