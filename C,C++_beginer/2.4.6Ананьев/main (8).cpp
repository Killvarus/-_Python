#include"Geom_fig.h"

int main()
{
	setlocale(LC_ALL, "Russian");
	pyramid A(5, 5);
	cylinder B(5, 5);
	geom_fig* geom[2] = { &A,&B };
	cout<<"Объем фигур в массиве: "<<sum_volume_func(geom, 2) << endl;

	return 0;
}