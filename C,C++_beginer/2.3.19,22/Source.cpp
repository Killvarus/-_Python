#include"Header.h"


int main() 
{
	Matrix  A, C, D, E;
	Vector V;
	cin >> A;
	//cin >> V;
	cin >> C;
	A= A * C;
	cout << A;
	return 0;
}