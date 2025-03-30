#include"Header.h"
int kol;
double prav,lev;
double hord(double(*pf)(double), double a, double b, double e)
{
    kol = 0;
   double c=0,c1;
   do 
    {
        kol++;
        c1 = c;
        c = b - pf(b) * ((b - a) / (pf(b) - pf(a)));
        
        if (pf(b) < 0 && pf(c) > 0 || pf(b) > 0 && pf(c) < 0)
        {
            a = c;
            lev = c;
            prav = c1;
        }
        else
        {
            b = c;
            lev = c1;
            prav = c;
        }
   } while (abs(c - c1) > e);

    return c;
}
double dichotomy(double (*pf)(double),double a, double b, double e)
{ 
    kol = 0;
    double c;
   do  {
        kol++;
        c = (a + b) / 2;
        if (pf(b) < 0 && pf(c) > 0 || pf(b) > 0 && pf(c) < 0)
        {
            a = c;
            lev = c;
            prav = b;
        }
        else
        {
            b = c;
            lev = a;
            prav = c;
        }
        
   } while (abs(lev - prav) > e);
 
    return ((a + b) / 2);
}
/*double kasat(double(*pf)(double), double(*pf1)(double), double a, double b, double e)
{
    lev = 0;
    prav = 0;
    kol = 0;
    double c=0,c1;
    double k = pf(b);
    int i = 0;
    do {
        if ((i == 0) && (k < 0)) {
            do {
                
                b = b + 0.1;
                k = pf(b);

            } while (k < 0);
        }
        i++;
        b = b - 0.1;
        k = pf(b);

        if (i == 10000) break; 
    } while (k > 0);
    do
    {
        c1 = c;
        c = b - (pf(b) / pf1(b));
        kol++;
        if (c1 > c) 
        {
            lev = c;
            prav = c1;
        }
        else
        {
            lev = c1;
            prav = c;
        }



    } while (abs(c - c1) > e);
    return(c);
}*/
double iter(double(*pf)(double), double a, double b, double e)
{
    double x=0,x1;
    prav = 0;
    kol = 0;
    do
    {
        kol++;
        x1 = x;
        x = pf(b);
        b = x;


    } while (abs(x - x1) > e);
    if (x1 > x)
    {
        lev = x;
        prav = x1;
    }
    else 
    {
        lev = x1;
        prav = x;
    }
    return(x);


}