#include"Header.h"
int srav, pere;
void Vstavka(TYPE *mass,int n)
{
    TYPE El;
    int place;
    pere = srav = 0;
    for (int i = 1; i < n; i++)
    {
        El = mass[i];
        place = i - 1;
        for (place = i - 1; place >= 0 && mass[place] > El; place--)
        {
            mass[place + 1] = mass[place];
            srav++;

        }
        mass[place + 1] = El;
        pere++;
    }
}
void Pyzir(TYPE* mass, int n)
{
    srav = pere = 0;
    for (int i = 0; i < n - 1; i++)
    {
        for (int k,j = (n- 1); j > i; j--)
        {
            srav++;
            if (mass[j - 1] > mass[j])
            {
               k = mass[j - 1]; 
                mass[j - 1] = mass[j];
                mass[j] = k;
                pere++;
            }
        }
    }
}
void Vibor(TYPE* mass, int n)
{
    pere = srav = 0;
    int min;
    TYPE t;
    for (int i = 0; i < n - 1; i++)
    {
        min = i;
        for (int j = i + 1; j < n; j++) 
        {
            if (mass[j] < mass[min])
            {
                min = j;
                pere++;
            }
            srav++;
                
        }
        t = mass[i];
        mass[i] = mass[min];
        mass[min] = t;
    }
}
void vivod(TYPE* mass, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("x[%d]=%lf \n", i, *(mass + i));
    }

}
