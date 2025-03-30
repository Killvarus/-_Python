#include<iostream>
using namespace std;

class geom_fig {
	double h,s;
	string str;
public:
	geom_fig(string str_n=0, double s_n = 0, double h_n = 0) : s(s_n), h(h_n), str(str_n)
	{
		cout << "Hello, geom_fig!" << endl;
	}
	~geom_fig()
	{
		cout << "geom_fig RIP" << endl;
	}
	void correct_title(string src)
	{
		str = src;
	}
	virtual double volume() = 0;
	double get_s() { return s; }
	double get_h() { return h; }
	friend ostream& operator<<(ostream& os, geom_fig& src);
	friend double sum_volume_func(geom_fig** src, int size);
};
double sum_volume_func(geom_fig** src, int size)
{
	double sum=0;
	for (int i = 0; i < size; i++)
	{
		sum += src[i]->volume();
	}
	return sum;
}
ostream& operator<<(ostream& os, geom_fig& src)
{
	os << "Геометрический объект " << src.str << " с площадью основания "<<src.s<<" и высотой "<<src.h << endl;
	os << "И объемом " << src.volume()<<endl;
	return os;
}
class pyramid : public geom_fig
{
public:
	pyramid(double s_n = 0, double h_n = 0,string str_n = "pyramid") : geom_fig(str_n, s_n, h_n)
	{
		cout << "Hello pyramid" << endl;
	}
	~pyramid()
	{
		cout << "pyramid RIP" << endl;
		cout << *this;

	}
	double volume()
	{
		return (1.0 / 3.0 * get_h() * get_s());
	}

};
class cylinder : public geom_fig
{
public:
	cylinder( double s_n = 0, double h_n = 0, string str_n = "cylinder") : geom_fig(str_n, s_n, h_n)
	{
		cout << "Hello cylinder" << endl;
	}
	~cylinder()
	{
		cout << "cylinder RIP" << endl;
		cout << *this;

	}
	double volume()
	{
		return (get_s()*get_h());
	}

};
