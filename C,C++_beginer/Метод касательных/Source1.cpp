#include"Header.h"
using namespace std;
int main() {

	double g, k, p, f, f1, t, t1, l;
	cout << ("Vvedite predpolozhitelnoe pogranichnoe znachenie argymenta ot '+' k '-': ");
	cin >> g;
	int i = 0;
	k = func(g);
	do {
		if ((i == 0) && (k < 0)) {
			do {
				g = g + 0.001;
				k = func(g);

			} while (k < 0);
		}
		i++;
		g = g - 0.001;
		k = func(g);

		if (i == 10000) { cout << "MNOGO!" << endl; break; }
	} while (k > 0);
	cout << k << setprecision(30) << "  Minimalnoe obnaruzhennoe polozhitelnoe znachenie" << endl << g << "  Sootvetstvyushee emy znachenie argumenta" << endl;
	t = g + 0.001;
	p = func(g);
	cout << p << "  Maximalnoe obnaruzhennoe otricatelnoe znachenie" << endl << t << "  Sootvetstvyushee emy znachenie argumenta" << endl;

	do {
		t1 = t - func(t) / func1(t);
		l = t;
		t = t1;

	} while (fabs(t - l) > 10e-35);
	cout << t << "  Priblizitelnoe znachenie resheniya yravneniya" << endl;


	return 0;
}