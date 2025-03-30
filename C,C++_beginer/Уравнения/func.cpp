#include"Header.h"
double tg_2x(double(x))
{
	return (tan(2 * x)-x-1);
	//return ( x*x - 1);
}
double tan_2x_pr(double x)
{
	return ((2 / (cos(2*x) * cos(2*x))) - 1);
}
double _exp_x(double x)
{
	return (2 * exp(-x) - x);
}
double _exp_x_pr(double x)
{
	return(-2 * exp(-x) - 1);

}
/*double exp_3x(double x)
{
	return (exp(-3 * x) - x);
}*/
double _exp_x1(double x)
{
	return (2 * exp(-x));
}
double x_2(double x)
{
	return (2 * x * x - 4 * x + 1);
}
double x_2_1(double x)
{
	return (2 * x * x - 3 * x + 1);
}
double i_iter(double x)//24
{
	return(exp(-pow(x,2)) + 2);
}
double i_iter1(double x)
{
return(exp(-pow(x, 2)) + 2-x);

}
double i_hord(double x)
{
	return tan(x / 2) / 10 - 1 - x;
	//return( exp(-pow(x, 3)) - 1 - pow(x,3));		
}//20
double i_dih(double x)//17
{
	return(exp(-3 * x) - x);
}
	