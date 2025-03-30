#include <string>
#include<iostream>
using namespace std;

class geom_obj {
	double x, y, z, dx, dy, dz;
public:
	geom_obj(double xi = 0, double yi = 0, double zi = 0, double dxi = 0, double dyi = 0, double dzi = 0) :x(xi), y(yi), z(zi), dx(dxi), dy(dyi), dz(dzi) {}
	virtual double volume()= 0;
	virtual string className() = 0;
	friend ostream& operator<<(ostream& os, geom_obj&);
};
ostream& operator<<(ostream& os, geom_obj& fig)
{
	os << fig.className() <<endl<< "Koordinati centra: (" << fig.x << ";" << fig.y << ";" << fig.z << ")";
	os << "Razmeri vdol osey: " << "oX: " << fig.dx << ", oY: " << fig.dy << ", oZ: " << fig.dz;
	return os;
}