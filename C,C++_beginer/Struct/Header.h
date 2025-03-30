#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include<math.h>
using namespace std;
int numberCount(FILE* input);

typedef struct {
	double* a, * b, * c;

}equat;
typedef struct 
{
	double real1,real2;
	double imag1,imag2;

}solv;
void solve(double a, double b, double c);
