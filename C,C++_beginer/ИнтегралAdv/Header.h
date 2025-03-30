#include<iostream>
#include<math.h>
#include<stdio.h>
using namespace std;
#define h ((b-a)/n)
double mrectangle (double (*pf)(double), int n, double a, double b);
double Parab (double(*pf)(double), int n, double a, double b);
double Trap (double(*pf)(double), int n, double a, double b); 
double sin_2 (double x);
double lrectangle (double (*pf)(double), int n, double a, double b);
double rrectangle (double (*pf)(double), int n, double a, double b);
double Tochnost (double(*pf2)(double), double(*pf1)(double (*pf2)(double), int, double, double), int n, double a, double b,double z);

