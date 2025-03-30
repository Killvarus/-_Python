#include "geom_obj.h"
class piramida : public geom_obj {
	piramida(double xi = 0, double yi = 0, double zi = 0, double dxi = 0, double dyi = 0, double dzi = 0) :geom_obj(xi, yi, zi, dxi, dyi, dzi) {}
	virtual double volume()
	{return }
};