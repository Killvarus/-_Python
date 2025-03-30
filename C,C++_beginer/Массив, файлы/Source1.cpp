#include "Header1.h"
int i, j, t, v, y;
void sort(int A[], int size) {

	y = 0;
	v = 0;
	for (i = 0; i < size; i++)
	{
		t = A[i];
		for (j = i - 1; j >= 0 && A[j] < t; j--) {
			A[j + 1] = A[j];
			y = y + 1;

		}
		v = v + 1;
		A[j + 1] = t;
	}

}