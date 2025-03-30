#include"Header.h"

int main()
{
	srand(time(NULL));
	setlocale(LC_ALL, "Russian");
	float x, y;
	unsigned int count=0,count1=0;
	for (int i = 0; i < 1001; i++)
	{
		x = 0 + rand() % 10;
		y = 0 + rand() % 10;
		if ((x-5) * (x-5) + (y-5) * (y-5) <= 25) count++;
		else  count1++;
		if(i%100==0) 
		{
			printf("Число ПИ, вычисленное методом Монте-Карло для %d пар чисел: %lf\n",i, 4 * (double)count / ((double)count + (double)count1));
			printf("Погрешность для данного количества пар: %lf\n", abs(M_PI - 4 * (double)count / ((double)count + (double)count1)));
		}
		
	}
	


	return 0;
}