#include"Header.h"

int main()
{
	setlocale(LC_ALL, "Russian");
	TYPE* t,*z,*x;
	size_t N;
	unsigned int n;
	extern int pere, srav;
	printf("Введите размер создаваемого массива: ");
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
	printf("Изначальный массив 1: \n");
	vivod(t, n);
	printf("Изначальный массив 2: \n");
	vivod(z, n);
	printf("Изначальный массив 3: \n");
	vivod(x, n);
	printf("Массив 1, отсортированный методом Вставки: \n");
	Vstavka(t, n);
	vivod(t, n); 
	printf("Количество сравнений: %d\tКоличество перестановок: %d\n", srav, pere);
	printf("Массив 2, отсортированный методом пузырька: \n");
	Pyzir(z, n);
	vivod(z, n);
	printf("Количество сравнений: %d\tКоличество перестановок: %d\n", srav, pere);
	printf("Массив 3, отсортированный методом Выбора: \n");
	Vibor(x, n);
	vivod(x, n);
	printf("Количество сравнений: %d\tКоличество перестановок: %d\n", srav, pere);



	return 0;
}